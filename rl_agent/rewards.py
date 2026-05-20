"""
rewards.py
──────────
All reward shaping for the VJ RL agent.

Reward components (each returns a float, agent gets the sum):

  R_source       — penalise missing source, reward preemptive video switch
  R_flicker_low  — penalise rapid low-level actions (shader add/remove churn)
  R_flicker_high — penalise rapid high-level goal changes
  R_crowd        — reward visual energy matching crowd energy
  R_perf         — small penalty for dropped frames

Each component is normalised to roughly [-1, +1] per step before weighting.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from shared_types import EngineState


# ─────────────────────────────────────────────────────────────────────────────
#  Flicker tracker: exponential moving average of action rate
# ─────────────────────────────────────────────────────────────────────────────

class FlickerTracker:
    """
    Tracks how rapidly an agent is issuing structural changes
    (add/remove passes, switch source).

    action_rate is an EMA of "did we do a structural change this step?"
    Smooth change (1 action every ~10 steps) → rate ≈ 0.1
    Flickering (action every step) → rate ≈ 1.0
    """
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha        # EMA smoothing factor
        self.action_rate: float = 0.0

    def update(self, did_structural_change: bool) -> float:
        x = 1.0 if did_structural_change else 0.0
        self.action_rate = self.alpha * x + (1.0 - self.alpha) * self.action_rate
        return self.action_rate

    def penalty(self, threshold: float = 0.25) -> float:
        """
        Returns 0 when calm, increasingly negative as flicker rate rises above threshold.
        Shaped as a smooth cliff: penalty = -tanh((rate - threshold) * 6)
        """
        excess = self.action_rate - threshold
        if excess <= 0:
            return 0.0
        return -float(np.tanh(excess * 6.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Individual reward components
# ─────────────────────────────────────────────────────────────────────────────

def r_source(state: EngineState, prev_state: EngineState) -> float:
    """
    +1.0  if source is present
    -1.0  if source is missing (no video → penalise harshly)
    +0.5  bonus if we transitioned *away* from a near-end video
           (preemptive switch: agent didn't wait for hard EOF)
    -0.3  if video just hit near_end without a switch being ready
    """
    if not state.has_source:
        return -1.0

    reward = 1.0   # base: source present

    # Bonus: preemptive switch happened (prev was near_end, now new source loaded)
    prev_near = (
        prev_state.has_source and
        prev_state.source_dur_sec > 0 and
        prev_state.source_pos_sec / prev_state.source_dur_sec > 0.85
    )
    source_changed = state.source_path != prev_state.source_path
    if prev_near and source_changed:
        reward += 0.5

    # Mild warning: near end but nothing changed yet
    curr_near = (
        state.source_dur_sec > 0 and
        state.source_pos_sec / state.source_dur_sec > 0.85 and
        not source_changed
    )
    if curr_near:
        reward -= 0.3

    return float(np.clip(reward, -1.0, 1.5))


def r_crowd(state: EngineState) -> float:
    """
    Reward for matching visual intensity to crowd energy.
    Visual intensity proxy = active_passes / max_passes * average_shader_weight.
    We want: visual_intensity ≈ crowd.energy
    """
    from features import N_SHADERS
    visual_intensity = state.active_passes / max(N_SHADERS, 1)
    mismatch = abs(visual_intensity - state.energy)
    # Perfect match → 1.0; complete mismatch → -1.0
    return float(1.0 - 2.0 * mismatch)


def r_perf(state: EngineState, target_ms: float = 33.3) -> float:
    """Small penalty for frame drops. 0 at 60fps, -1 at 3× frametime."""
    ratio = state.frame_time_ms / target_ms
    if ratio <= 1.2:
        return 0.0
    return float(-np.tanh((ratio - 1.2) * 2.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Composite reward for each level
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    source:       float = 2.0   # continuity is the primary task
    flicker:      float = 1.5   # stability matters a lot
    crowd:        float = 0.5   # crowd matching is nice-to-have
    perf:         float = 0.3


DEFAULT_WEIGHTS = RewardWeights()


def compute_low_level_reward(
    state:     EngineState,
    prev_state: EngineState,
    flicker:   FlickerTracker,
    did_structural_change: bool,
    weights:   RewardWeights = DEFAULT_WEIGHTS,
) -> tuple[float, dict]:
    """
    Returns (total_reward, component_dict) for the low-level policy.
    component_dict is for logging/debugging.
    """
    flicker.update(did_structural_change)

    components = {
        "source":  r_source(state, prev_state)  * weights.source,
        "flicker": flicker.penalty()             * weights.flicker,
        "crowd":   r_crowd(state)                * weights.crowd,
        "perf":    r_perf(state)                 * weights.perf,
    }
    total = sum(components.values())
    return total, components


def compute_high_level_reward(
    state:      EngineState,
    prev_state: EngineState,
    flicker:    FlickerTracker,
    did_goal_change: bool,
    weights:    RewardWeights = DEFAULT_WEIGHTS,
) -> tuple[float, dict]:
    """
    High-level policy cares more about source continuity and less about
    per-frame crowd matching (that's the low-level's job).
    """
    flicker.update(did_goal_change)

    components = {
        "source":  r_source(state, prev_state)  * weights.source,
        "flicker": flicker.penalty(threshold=0.1) * weights.flicker,  # stricter
        "crowd":   r_crowd(state)                * (weights.crowd * 0.3),
        "perf":    r_perf(state)                 * weights.perf,
    }
    total = sum(components.values())
    return total, components
