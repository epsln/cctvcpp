"""
high_level_agent.py
───────────────────
The high-level policy operates at ~0.1 Hz (every 10 s).
It sets *goals* for the low-level rather than issuing engine commands directly.

Action space (discrete):
  0  HOLD         — keep the current goal
  1  CALM         — low intensity, minimal shaders, crowd energy ≈ 0.0–0.3
  2  MEDIUM       — moderate intensity, 2-3 shaders
  3  HIGH_ENERGY  — max intensity, all shaders, match peak crowd
  4  TRANSITION   — signal that a new video should be loaded soon
  5  GLITCH_MODE  — emphasise glitch + chromatic (high excitement)
  6  AMBIENT_MODE — kaleidoscope + feedback + color_grade (calm/chill)

The goal is passed to the low-level as a dict.
The high-level's reward is the same total reward but with stricter flicker penalty.
"""

from __future__ import annotations
import time, os
import numpy as np

from shared_types import EngineState
from features import HistoryBuffer, HIGH_LEVEL_DIM
from rewards import FlickerTracker, compute_high_level_reward
from policy import MLPActorCritic

# ─────────────────────────────────────────────────────────────────────────────

HOLD         = 0
CALM         = 1
MEDIUM       = 2
HIGH_ENERGY  = 3
TRANSITION   = 4
GLITCH_MODE  = 5
AMBIENT_MODE = 6
N_ACTIONS    = 7

GOAL_DEFS = {
    HOLD:        {"target_intensity": None, "preferred_shaders": None, "transition": False},
    CALM:        {"target_intensity": 0.1,  "preferred_shaders": ["color_grade"],       "transition": False},
    MEDIUM:      {"target_intensity": 0.4,  "preferred_shaders": ["color_grade", "bloom"],  "transition": False},
    HIGH_ENERGY: {"target_intensity": 0.9,  "preferred_shaders": ["glitch", "chromatic", "bloom"], "transition": False},
    TRANSITION:  {"target_intensity": 0.5,  "preferred_shaders": ["color_grade"],       "transition": True},
    GLITCH_MODE: {"target_intensity": 0.8,  "preferred_shaders": ["glitch", "chromatic", "feedback"], "transition": False},
    AMBIENT_MODE:{"target_intensity": 0.3,  "preferred_shaders": ["kaleidoscope", "feedback", "color_grade"], "transition": False},
}


class HighLevelAgent:
    def __init__(
        self,
        weights_path:  str = "checkpoints/high_level",
        hidden_dim:    int = 64,
        step_interval: float = 10.0,  # seconds between decisions
    ):
        self.weights_path  = weights_path
        self.step_interval = step_interval
        self._last_step_t  = 0.0
        self._current_goal = GOAL_DEFS[CALM].copy()
        self._current_action = CALM

        self.history = HistoryBuffer()

        self.policy = MLPActorCritic(
            input_dim  = HIGH_LEVEL_DIM,
            n_actions  = N_ACTIONS,
            hidden_dim = hidden_dim,
            name       = "high_level",
        )
        self.policy.load(weights_path)

        self.flicker = FlickerTracker(alpha=0.05)   # very slow decay for high level

        self._prev_obs:    np.ndarray  = self.history.as_vector()
        self._prev_action: int         = CALM
        self._prev_state:  EngineState = EngineState()
        self.step_n = 0
        self.total_reward = 0.0

    def should_step(self) -> bool:
        return (time.time() - self._last_step_t) >= self.step_interval

    def observe(self, state: EngineState) -> None:
        """Called every low-level tick to keep history buffer fresh."""
        self.history.push(state)

    def current_goal(self) -> dict:
        return self._current_goal.copy()

    def step(self, state: EngineState) -> dict:
        """
        Make a high-level decision.
        Returns the new goal dict to pass to the low-level agent.
        """
        obs = self.history.as_vector()

        # ── RL update on previous high-level transition ──
        if self.step_n > 0:
            reward, components = compute_high_level_reward(
                state          = state,
                prev_state     = self._prev_state,
                flicker        = self.flicker,
                did_goal_change = (self._prev_action != self._current_action),
            )
            self.total_reward += reward
            train_info = self.policy.update(
                obs      = self._prev_obs,
                action   = self._prev_action,
                reward   = reward,
                next_obs = obs,
                done     = False,
            )
            goal_names = {0:"HOLD",1:"CALM",2:"MEDIUM",3:"HIGH",4:"TRANSIT",5:"GLITCH",6:"AMBIENT"}
            print(f"[HighLevel step {self.step_n}] "
                  f"goal={goal_names.get(self._current_action,'?')} "
                  f"reward={reward:.3f} "
                  f"(src={components['source']:.2f} "
                  f"flick={components['flicker']:.2f}) "
                  f"adv={train_info['advantage']:.3f}")

            if self.step_n % 50 == 0:
                self.policy.save(self.weights_path)

        # ── Choose action ──────────────────────────────────
        # Bias HOLD action when crowd energy is ambiguous (reduce unnecessary changes)
        action, log_p, value = self.policy.act(obs)

        # Override: if crowd energy is very high and we're in CALM → force at least MEDIUM
        if state.energy > 0.8 and action == CALM:
            action = MEDIUM

        goal = GOAL_DEFS[action].copy()

        # Fill HOLD with the previous goal's details
        if action == HOLD:
            goal = self._current_goal.copy()
            goal["transition"] = False

        self._current_goal   = goal
        self._current_action = action
        self._prev_obs       = obs
        self._prev_action    = action
        self._prev_state     = state
        self._last_step_t    = time.time()
        self.step_n         += 1

        return goal
