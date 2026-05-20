"""
networks.py
───────────
PyTorch actor-critic networks for both hierarchy levels.

Architecture overview
─────────────────────

                        ┌─────────────────────────────┐
  visual obs (17)  ───► │   Visual encoder (MLP)       │
                        │   17 → 64 → 64               │
                        └──────────┬──────────────────┘
                                   │ h_visual (64)
                        ┌──────────▼──────────────────┐
  audio feat (138) ───► │   Audio encoder (MLP)        │
                        │   138 → 128 → 64             │
                        └──────────┬──────────────────┘
                                   │ h_audio (64)
                        ┌──────────▼──────────────────┐
                        │   Fusion (concat + Linear)   │
                        │   128 → 128                  │
                        └──────────┬──────────────────┘
                                   │ h_fused (128)
                     ┌─────────────┴──────────────┐
                     ▼                             ▼
             Actor head                     Critic head
             128 → n_actions                128 → 1
             (softmax)                      (scalar value)

High-level policy:
  Visual input is a flattened history window (17 × 8 = 136)
  Audio input: same 138-dim vector
  Otherwise identical architecture.

Training:
  PPO-clip with generalised advantage estimation (GAE).
  We accumulate a rollout buffer of T steps before each update,
  then run K epochs of minibatch gradient descent.
  This is more stable than the per-step REINFORCE used before.
"""

from __future__ import annotations
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def mlp(dims: list[int], activation=nn.ELU, output_activation=None) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class AudioEncoder(nn.Module):
    """Encodes raw audio feature vector (mel + scalars) into a fixed embedding."""
    def __init__(self, input_dim: int = 138, embed_dim: int = 64):
        super().__init__()
        self.net = mlp([input_dim, 128, embed_dim], activation=nn.ELU)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.elu(self.net(x)))


class VisualEncoder(nn.Module):
    """Encodes visual pipeline state (pass bits + crowd + timing)."""
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = mlp([input_dim, 64, embed_dim], activation=nn.ELU)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.elu(self.net(x)))


class FusionTrunk(nn.Module):
    """Fuses visual and audio embeddings into a joint representation."""
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp([embed_dim * 2, hidden_dim, hidden_dim], activation=nn.ELU)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_vis: torch.Tensor, h_aud: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([h_vis, h_aud], dim=-1)
        return self.norm(F.elu(self.net(fused)))


# ─────────────────────────────────────────────────────────────────────────────
#  Full actor-critic
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(
        self,
        visual_dim:  int,
        audio_dim:   int,
        n_actions:   int,
        embed_dim:   int = 64,
        hidden_dim:  int = 128,
    ):
        super().__init__()
        self.visual_enc = VisualEncoder(visual_dim, embed_dim)
        self.audio_enc  = AudioEncoder(audio_dim,  embed_dim)
        self.trunk      = FusionTrunk(embed_dim, hidden_dim)

        self.actor_head  = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for actor head (avoids premature entropy collapse)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, vis_obs: torch.Tensor, aud_obs: torch.Tensor):
        h_vis  = self.visual_enc(vis_obs)
        h_aud  = self.audio_enc(aud_obs)
        trunk  = self.trunk(h_vis, h_aud)
        logits = self.actor_head(trunk)
        value  = self.critic_head(trunk).squeeze(-1)
        return logits, value

    def act(self, vis_obs: torch.Tensor, aud_obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action.
        Returns (action, log_prob, value, entropy).
        All inputs/outputs are on the same device as the model.
        """
        logits, value = self.forward(vis_obs, aud_obs)
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, vis_obs: torch.Tensor, aud_obs: torch.Tensor, actions: torch.Tensor):
        """For PPO update: recompute log_probs and values for stored transitions."""
        logits, values = self.forward(vis_obs, aud_obs)
        dist    = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


# ─────────────────────────────────────────────────────────────────────────────
#  PPO rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-size circular buffer for PPO.
    Stores (vis_obs, aud_obs, action, log_prob, reward, value, done).
    """

    def __init__(self, capacity: int, visual_dim: int, audio_dim: int, device: str = "cpu"):
        self.capacity   = capacity
        self.device     = device
        self.visual_dim = visual_dim
        self.audio_dim  = audio_dim
        self.reset()

    def reset(self):
        C, V, A = self.capacity, self.visual_dim, self.audio_dim
        self.vis_obs   = torch.zeros(C, V)
        self.aud_obs   = torch.zeros(C, A)
        self.actions   = torch.zeros(C, dtype=torch.long)
        self.log_probs = torch.zeros(C)
        self.rewards   = torch.zeros(C)
        self.values    = torch.zeros(C)
        self.dones     = torch.zeros(C)
        self.ptr = 0
        self.full = False

    def push(self, vis_obs, aud_obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.vis_obs[i]   = vis_obs.detach().cpu() if isinstance(vis_obs, torch.Tensor) else torch.tensor(vis_obs)
        self.aud_obs[i]   = aud_obs.detach().cpu() if isinstance(aud_obs, torch.Tensor) else torch.tensor(aud_obs)
        self.actions[i]   = int(action)
        self.log_probs[i] = float(log_prob)
        self.rewards[i]   = float(reward)
        self.values[i]    = float(value)
        self.dones[i]     = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def is_ready(self) -> bool:
        return self.full or self.ptr >= self.capacity // 2

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """
        Generalised Advantage Estimation.
        Returns (advantages, returns) tensors of length == stored transitions.
        """
        n = self.capacity if self.full else self.ptr
        advantages = torch.zeros(n)
        last_gae   = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else float(self.values[t + 1])
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = float(self.rewards[t]) + gamma * next_val * next_non_terminal - float(self.values[t])
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        return advantages, returns

    def get(self, last_value: float = 0.0):
        """Returns dict of tensors ready for PPO update."""
        n = self.capacity if self.full else self.ptr
        adv, ret = self.compute_gae(last_value)
        # Normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "vis_obs":   self.vis_obs[:n],
            "aud_obs":   self.aud_obs[:n],
            "actions":   self.actions[:n],
            "log_probs": self.log_probs[:n],
            "advantages": adv,
            "returns":   ret,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  PPO trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Wraps ActorCritic + RolloutBuffer + PPO update loop.
    Designed to be called online: collect steps, then call update() when buffer ready.
    """

    def __init__(
        self,
        visual_dim:   int,
        audio_dim:    int,
        n_actions:    int,
        rollout_steps: int = 64,
        ppo_epochs:    int = 4,
        minibatch_size: int = 16,
        lr:            float = 3e-4,
        clip_eps:      float = 0.2,
        vf_coef:       float = 0.5,
        ent_coef:      float = 0.01,
        gamma:         float = 0.99,
        gae_lam:       float = 0.95,
        grad_clip:     float = 0.5,
        device:        str   = "cpu",
        name:          str   = "agent",
    ):
        self.name          = name
        self.rollout_steps = rollout_steps
        self.ppo_epochs    = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.gamma         = gamma
        self.gae_lam       = gae_lam
        self.grad_clip     = grad_clip
        self.device        = torch.device(device)

        self.net = ActorCritic(visual_dim, audio_dim, n_actions).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_steps, visual_dim, audio_dim, device)

        self.step_count   = 0
        self.update_count = 0
        self._last_update_info: dict = {}

    # ── Step interface ───────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, vis_obs: np.ndarray, aud_obs: np.ndarray) -> tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        v = torch.tensor(vis_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.tensor(aud_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob, value, _ = self.net.act(v, a)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store(self, vis_obs, aud_obs, action, log_prob, reward, value, done):
        self.buffer.push(vis_obs, aud_obs, action, log_prob, reward, value, done)
        self.step_count += 1

    def should_update(self) -> bool:
        return self.buffer.is_ready()

    def update(self, last_vis_obs: np.ndarray, last_aud_obs: np.ndarray) -> dict:
        """Run PPO update epochs over the current buffer."""
        # Bootstrap value for GAE
        with torch.no_grad():
            v = torch.tensor(last_vis_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            a = torch.tensor(last_aud_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, last_value, _, _ = self.net.act(v, a)
        last_value = float(last_value)

        batch = self.buffer.get(last_value)
        # Move to device
        vis_obs   = batch["vis_obs"].to(self.device)
        aud_obs   = batch["aud_obs"].to(self.device)
        actions   = batch["actions"].to(self.device)
        old_lps   = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns   = batch["returns"].to(self.device)

        n = vis_obs.shape[0]
        total_pl, total_vl, total_el = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            # Random minibatch indices
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start: start + self.minibatch_size]
                if len(mb) < 2:
                    continue

                new_lps, values, entropy = self.net.evaluate(
                    vis_obs[mb], aud_obs[mb], actions[mb]
                )

                # PPO clipped policy loss
                ratio = torch.exp(new_lps - old_lps[mb])
                adv_mb = advantages[mb]
                pl1 = ratio * adv_mb
                pl2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(pl1, pl2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, returns[mb])

                # Entropy bonus (exploration)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()

                total_pl += policy_loss.detach().item()
                total_vl += value_loss.detach().item()
                total_el += (-entropy_loss).detach().item()
                n_updates += 1

        self.buffer.reset()
        self.update_count += 1
        info = {
            "policy_loss": total_pl / max(n_updates, 1),
            "value_loss":  total_vl / max(n_updates, 1),
            "entropy":     total_el / max(n_updates, 1),
            "update":      self.update_count,
            "steps":       self.step_count,
        }
        self._last_update_info = info
        return info

    # ── Save / load ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "net":           self.net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "step_count":    self.step_count,
            "update_count":  self.update_count,
        }, path + ".pt")

    def load(self, path: str) -> bool:
        pt = path + ".pt"
        if not os.path.exists(pt):
            return False
        ckpt = torch.load(pt, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count   = ckpt.get("step_count", 0)
        self.update_count = ckpt.get("update_count", 0)
        print(f"[{self.name}] Loaded checkpoint: {pt} "
              f"(step={self.step_count}, update={self.update_count})")
        return True
