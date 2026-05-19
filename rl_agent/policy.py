"""
policy.py
─────────
Lightweight numpy-only MLP policy for both hierarchy levels.

No PyTorch / TF dependency — runs on any machine that has numpy.
Uses a simple REINFORCE (policy gradient) update with a value baseline
(actor-critic variant: separate value head, shared body).

Architecture:
  input → Linear(hidden) → ReLU → Linear(hidden) → ReLU → [policy_head, value_head]

  policy_head → softmax over discrete actions
  value_head  → scalar baseline

Training:
  - Online, one step at a time (no replay buffer needed for this scale)
  - Separate learning rates for actor and critic
  - Gradient clipping by norm
"""

from __future__ import annotations
import numpy as np
import json
import os
from typing import Optional


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
#  MLP with actor-critic heads
# ─────────────────────────────────────────────────────────────────────────────

class MLPActorCritic:
    """
    Shared-trunk actor-critic MLP.
    All weights stored as plain numpy arrays for portability.
    """

    def __init__(
        self,
        input_dim:  int,
        n_actions:  int,
        hidden_dim: int = 64,
        lr_actor:   float = 3e-4,
        lr_critic:  float = 1e-3,
        gamma:      float = 0.97,
        name:       str   = "policy",
    ):
        self.input_dim  = input_dim
        self.n_actions  = n_actions
        self.hidden_dim = hidden_dim
        self.lr_actor   = lr_actor
        self.lr_critic  = lr_critic
        self.gamma      = gamma
        self.name       = name

        # ── Layer weights (He init) ──────────────
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim,  hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Actor head (policy logits)
        self.Wa = np.random.randn(hidden_dim, n_actions).astype(np.float32) * 0.01
        self.ba = np.zeros(n_actions, dtype=np.float32)

        # Critic head (state value)
        self.Wv = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.01
        self.bv = np.zeros(1, dtype=np.float32)

        # Training state
        self.step_count = 0

        # Adam optimizer state
        self._init_adam()

    # ── Forward pass ────────────────────────────

    def _forward(self, obs: np.ndarray):
        """Returns (h2, logits, value, probs) — keeps activations for backward."""
        h1 = relu(obs @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.Wa + self.ba
        value  = float((h2 @ self.Wv + self.bv)[0])
        probs  = softmax(logits)
        return h1, h2, logits, value, probs

    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        """
        Sample an action.
        Returns (action_index, log_prob, value_estimate).
        """
        _, _, _, value, probs = self._forward(obs)
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, value

    def greedy_act(self, obs: np.ndarray) -> int:
        """Argmax action (for evaluation / logging)."""
        _, _, _, _, probs = self._forward(obs)
        return int(np.argmax(probs))

    # ── REINFORCE + baseline update (one transition) ──

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> dict:
        """
        Actor-critic TD(0) update.
        Returns dict of scalars for logging.
        """
        # Forward current and next
        h1, h2, logits, value, probs = self._forward(obs)
        _, _, _, next_value, _       = self._forward(next_obs)

        # TD target and advantage
        target    = reward + (0.0 if done else self.gamma * next_value)
        advantage = target - value

        # ── Critic loss gradient (MSE) ───────────
        d_value   = -2.0 * advantage   # d(MSE)/d(value)
        d_Wv      = h2[:, None] * d_value
        d_bv      = np.array([d_value])

        d_h2_crit = (self.Wv * d_value).flatten()

        # ── Actor loss gradient (REINFORCE) ──────
        # ∇ log π(a|s) = one_hot(a) - probs
        one_hot   = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        d_logits  = -(one_hot - probs) * advantage  # negative for gradient ascent via descent

        d_Wa = h2[:, None] * d_logits[None, :]
        d_ba = d_logits

        d_h2_act  = self.Wa @ d_logits

        # Combine h2 gradients
        d_h2 = d_h2_crit + d_h2_act
        d_h2 *= (h2 > 0).astype(np.float32)  # ReLU mask

        d_W2 = h1[:, None] * d_h2[None, :]
        d_b2 = d_h2

        d_h1 = (self.W2 @ d_h2)
        d_h1 *= (h1 > 0).astype(np.float32)

        d_W1 = obs[:, None] * d_h1[None, :]
        d_b1 = d_h1

        # Gradient clipping (by global norm)
        grads = [d_W1, d_b1, d_W2, d_b2, d_Wa, d_ba, d_Wv, d_bv]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        clip = 1.0
        if total_norm > clip:
            scale_g = clip / (total_norm + 1e-8)
            grads = [g * scale_g for g in grads]
        d_W1, d_b1, d_W2, d_b2, d_Wa, d_ba, d_Wv, d_bv = grads

        # Adam update
        self.step_count += 1
        self.W1 = self._adam_step("W1", self.W1, d_W1, self.lr_actor)
        self.b1 = self._adam_step("b1", self.b1, d_b1, self.lr_actor)
        self.W2 = self._adam_step("W2", self.W2, d_W2, self.lr_actor)
        self.b2 = self._adam_step("b2", self.b2, d_b2, self.lr_actor)
        self.Wa = self._adam_step("Wa", self.Wa, d_Wa, self.lr_actor)
        self.ba = self._adam_step("ba", self.ba, d_ba, self.lr_actor)
        self.Wv = self._adam_step("Wv", self.Wv, d_Wv, self.lr_critic)
        self.bv = self._adam_step("bv", self.bv, d_bv, self.lr_critic)

        return {
            "value":     value,
            "target":    target,
            "advantage": advantage,
            "grad_norm": float(total_norm),
        }

    # ── Adam ────────────────────────────────────

    def _init_adam(self):
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._b1_adam = 0.9
        self._b2_adam = 0.999
        self._eps_adam = 1e-8

    def _adam_step(self, key: str, param: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        if key not in self._m:
            self._m[key] = np.zeros_like(param)
            self._v[key] = np.zeros_like(param)
        m = self._m[key]
        v = self._v[key]
        m = self._b1_adam * m + (1.0 - self._b1_adam) * grad
        v = self._b2_adam * v + (1.0 - self._b2_adam) * grad**2
        self._m[key] = m
        self._v[key] = v
        m_hat = m / (1.0 - self._b1_adam ** self.step_count)
        v_hat = v / (1.0 - self._b2_adam ** self.step_count)
        return (param - lr * m_hat / (np.sqrt(v_hat) + self._eps_adam)).astype(np.float32)

    # ── Save / load ──────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            Wa=self.Wa, ba=self.ba,
            Wv=self.Wv, bv=self.bv,
            step_count=np.array([self.step_count]),
        )

    def load(self, path: str) -> bool:
        npz = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(npz):
            return False
        d = np.load(npz)
        self.W1 = d["W1"]; self.b1 = d["b1"]
        self.W2 = d["W2"]; self.b2 = d["b2"]
        self.Wa = d["Wa"]; self.ba = d["ba"]
        self.Wv = d["Wv"]; self.bv = d["bv"]
        self.step_count = int(d["step_count"][0])
        self._init_adam()   # reset optimizer state (weights carry over, momentum doesn't)
        return True
