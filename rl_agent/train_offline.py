"""
train_offline.py
────────────────
Offline pre-training harness using a simulated environment.

Running the RL agent against the live C++ engine is slow for early learning
(the engine runs at ~60 fps, agent steps are ~0.5 s, so ~120 agent steps/min).
This simulator lets us run thousands of episodes quickly to bootstrap the
weights before connecting to the real engine.

The simulator models:
  - A video source that runs for a random duration (30–300 s)
  - Crowd energy following a smooth sine + noise trajectory
  - Pipeline state updated according to agent commands

Usage:
  python3 train_offline.py --episodes 500 --steps_per_ep 200
"""

import argparse
import os
import random
import math
import numpy as np

from shared_types import EngineState
from features    import KNOWN_SHADERS
from rewards     import FlickerTracker, compute_low_level_reward, compute_high_level_reward
from policy      import MLPActorCritic
from features    import base_features, HistoryBuffer, BASE_DIM, HIGH_LEVEL_DIM
from low_level_agent  import LowLevelAgent, N_ACTIONS as LOW_N, NOOP, _is_structural
from high_level_agent import HighLevelAgent, N_ACTIONS as HIGH_N, GOAL_DEFS


# ─────────────────────────────────────────────────────────────────────────────
#  Simulated environment
# ─────────────────────────────────────────────────────────────────────────────

class SimEnv:
    """
    Fast numpy simulation of the VJ engine state.
    Does not produce pixels — only tracks pipeline state and timing.
    """

    VIDEOS = [f"video_{i:02d}.mp4" for i in range(10)]

    def __init__(self):
        self.reset()

    def reset(self) -> EngineState:
        self.t = 0.0
        self.dt = 0.5          # low-level step interval

        # Video
        self.video_idx     = random.randint(0, len(self.VIDEOS) - 1)
        self.video_dur     = random.uniform(30.0, 180.0)
        self.video_pos     = random.uniform(0.0, self.video_dur * 0.5)
        self.has_source    = random.random() > 0.1  # 10% chance of starting without source
        self.source_path   = self.VIDEOS[self.video_idx] if self.has_source else ""

        # Crowd trajectory
        self._crowd_phase  = random.uniform(0, 2 * math.pi)
        self._crowd_speed  = random.uniform(0.05, 0.3)

        # Pipeline
        self.active_shaders: set[str] = set()
        self.frame_time_ms = 16.7

        return self._make_state()

    def _crowd_energy(self) -> float:
        base = 0.5 + 0.4 * math.sin(self.t * self._crowd_speed + self._crowd_phase)
        noise = random.gauss(0, 0.05)
        return float(np.clip(base + noise, 0.0, 1.0))

    def _make_state(self) -> EngineState:
        energy = self._crowd_energy()
        s = EngineState(
            timestamp       = self.t,
            frame_number    = int(self.t / self.dt),
            frame_time_ms   = self.frame_time_ms,
            has_source      = self.has_source,
            source_path     = self.source_path,
            source_pos_sec  = self.video_pos,
            source_dur_sec  = self.video_dur if self.has_source else 0.0,
            source_near_end = (self.video_pos / self.video_dur > 0.85) if self.has_source else False,
            active_passes   = len(self.active_shaders),
            passes          = [{"id": f"rl_{s}", "shader": s, "enabled": True}
                               for s in self.active_shaders],
            energy    = energy,
            density   = float(np.clip(energy * 0.8 + random.gauss(0, 0.1), 0, 1)),
            pulse     = 1.0 if (self.t % 0.5) < 0.05 else 0.0,
            frequency = float(np.clip(0.5 + 0.4 * math.sin(self.t * 2.1), 0, 1)),
            sentiment = math.sin(self.t * 0.1),
        )
        return s

    def step(self, commands: list[dict]) -> EngineState:
        """Apply commands, advance time, return new state."""
        for cmd in commands:
            t = cmd.get("type", "")
            if t == "ADD_PASS":
                shader = cmd.get("shader_name", "")
                if shader in KNOWN_SHADERS:
                    self.active_shaders.add(shader)
            elif t == "REMOVE_PASS":
                pid = cmd.get("pass_id", "")
                shader = pid.replace("rl_", "")
                self.active_shaders.discard(shader)
            elif t == "SET_SOURCE":
                path = cmd.get("source_path", "")
                if path:
                    self.has_source   = True
                    self.source_path  = path
                    self.video_pos    = 0.0
                    self.video_dur    = random.uniform(30.0, 180.0)
                    self.video_idx    = (self.video_idx + 1) % len(self.VIDEOS)
            # ENABLE_PASS, SET_UNIFORM — update tracking
            elif t == "ENABLE_PASS":
                pid     = cmd.get("pass_id", "")
                shader  = pid.replace("rl_", "")
                enabled = cmd.get("enabled", True)
                if enabled:  self.active_shaders.add(shader)
                else:        self.active_shaders.discard(shader)

        # Advance time
        self.t        += self.dt
        self.video_pos = min(self.video_pos + self.dt, self.video_dur)

        # Video ended → go dark
        if self.video_pos >= self.video_dur:
            self.has_source  = False
            self.source_path = ""

        # Simulate occasional frame drop
        self.frame_time_ms = random.gauss(16.7, 2.0)

        return self._make_state()


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_low_level(episodes: int, steps_per_ep: int, save_path: str):
    env     = SimEnv()
    policy  = MLPActorCritic(input_dim=BASE_DIM, n_actions=LOW_N, name="low_level")
    policy.load(save_path)
    flicker = FlickerTracker(alpha=0.2)

    ep_rewards = []
    for ep in range(episodes):
        state       = env.reset()
        flicker     = FlickerTracker(alpha=0.2)
        ep_reward   = 0.0
        prev_state  = state
        last_action = None

        for step in range(steps_per_ep):
            obs    = base_features(state)
            action, _, _ = policy.act(obs)
            if action is not last_action:
                state.last_ll_t = 0
            state.last_ll_t += step
            last_action = action

            # Translate action to command list (same logic as LowLevelAgent)
            _dummy = LowLevelAgent.__new__(LowLevelAgent)
            _dummy._intensity  = 0.5
            _dummy.library     = SimEnv.VIDEOS
            _dummy.lib_idx     = 0
            cmds = _dummy._action_to_commands(action, state)
            cmd_dicts = [c.to_dict() for c in cmds]

            next_state = env.step(cmd_dicts)
            next_state.last_ll_t = state.last_ll_t
            next_obs   = base_features(next_state)

            reward, _ = compute_low_level_reward(
                state      = next_state,
                prev_state = state,
                flicker    = flicker,
                did_structural_change = _is_structural(action),
            )
            ep_reward += reward

            policy.update(obs, action, reward, next_obs, done=(step == steps_per_ep - 1))

            prev_state = state
            state      = next_state


        ep_rewards.append(ep_reward)
        if ep % 50 == 0:
            avg = sum(ep_rewards[-50:]) / min(len(ep_rewards), 50)
            print(f"[LowLevel] ep={ep:4d}  avg_reward={avg:8.2f}  "
                  f"steps={policy.step_count}")
            policy.save(save_path)

    policy.save(save_path)
    print(f"[LowLevel] Training complete. Final avg reward: {sum(ep_rewards[-100:])/100:.2f}")


def train_high_level(episodes: int, steps_per_ep: int, save_path: str):
    from high_level_agent import GOAL_DEFS, HOLD, N_ACTIONS as HIGH_N
    env     = SimEnv()
    history = HistoryBuffer()
    policy  = MLPActorCritic(input_dim=HIGH_LEVEL_DIM, n_actions=HIGH_N, name="high_level")
    policy.load(save_path)
    flicker = FlickerTracker(alpha=0.05)

    ep_rewards = []
    t_change = -1 
    last_action = None
    for ep in range(episodes):
        state        = env.reset()
        history      = HistoryBuffer()
        flicker      = FlickerTracker(alpha=0.05)
        ep_reward    = 0.0
        prev_state   = state
        prev_action  = HOLD

        for step in range(steps_per_ep):
            history.push(state)
            obs    = history.as_vector()
            action, _, _ = policy.act(obs)
            if action is not last_action:
                state.last_hl_t = 0
            last_action = action
            state.last_hl_t += step

            # High-level: no direct commands to env, just advance env a few low steps
            for _ in range(20):   # ~10s of low-level steps
                state = env.step([])

            history.push(state)
            next_obs = history.as_vector()

            reward, _ = compute_high_level_reward(
                state          = state,
                prev_state     = prev_state,
                flicker        = flicker,
                did_goal_change = (action != prev_action),
            )
            ep_reward += reward

            policy.update(obs, action, reward, next_obs, done=(step == steps_per_ep - 1))

            prev_state  = state
            prev_action = action

        ep_rewards.append(ep_reward)
        if ep % 50 == 0:
            avg = sum(ep_rewards[-50:]) / min(len(ep_rewards), 50)
            print(f"[HighLevel] ep={ep:4d}  avg_reward={avg:8.2f}  "
                  f"steps={policy.step_count}")
            policy.save(save_path)

    policy.save(save_path)
    print(f"[HighLevel] Training complete. Final avg reward: {sum(ep_rewards[-100:])/100:.2f}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",     type=int, default=300)
    ap.add_argument("--steps_per_ep", type=int, default=150)
    ap.add_argument("--checkpoints",  default="checkpoints")
    ap.add_argument("--level",        choices=["low","high","both"], default="both")
    args = ap.parse_args()

    os.makedirs(args.checkpoints, exist_ok=True)
    np.random.seed(42)
    random.seed(42)

    if args.level in ("low", "both"):
        print("=" * 50)
        print("Training LOW-LEVEL policy")
        print("=" * 50)
        train_low_level(
            args.episodes, args.steps_per_ep,
            os.path.join(args.checkpoints, "low_level"),
        )

    if args.level in ("high", "both"):
        print("=" * 50)
        print("Training HIGH-LEVEL policy")
        print("=" * 50)
        train_high_level(
            args.episodes, args.steps_per_ep,
            os.path.join(args.checkpoints, "high_level"),
        )


if __name__ == "__main__":
    main()
