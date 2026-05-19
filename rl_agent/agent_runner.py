"""
agent_runner.py
───────────────
Top-level runner for the hierarchical VJ RL agent.

Usage:
  python3 agent_runner.py --media_dir /path/to/videos \
                          --state_json vj_state.json  \
                          --cmd_json   vj_commands.json

The runner:
  1. Reads vj_state.json at ~10 Hz
  2. Every 10 s  → high-level policy decides the goal
  3. Every 0.5 s → low-level policy issues engine commands
  4. Commands are written to vj_commands.json (atomic)
  5. Weights are saved to checkpoints/ periodically

Designed to run *alongside* the C++ engine process.
The two processes communicate only through the JSON files.
"""

import argparse
import time
import os
import json
import sys

from shared_types import EngineState, CommandBatch, atomic_write_json, read_json_safe
from high_level_agent import HighLevelAgent
from low_level_agent import LowLevelAgent


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--media_dir",  default="assets",          help="Directory with video/image files")
    p.add_argument("--state_json", default="vj_state.json",   help="Path engine writes state to")
    p.add_argument("--cmd_json",   default="vj_commands.json",help="Path agent writes commands to")
    p.add_argument("--checkpoints",default="checkpoints",     help="Directory for weight saves")
    p.add_argument("--no_train",   action="store_true",        help="Disable weight updates (eval mode)")
    p.add_argument("--poll_hz",    type=float, default=20.0,  help="How often to poll state file (Hz)")
    return p.parse_args()


def read_state(path: str) -> EngineState | None:
    d = read_json_safe(path)
    if d is None:
        return None
    return EngineState.from_dict(d)


def write_commands(path: str, batch: CommandBatch) -> None:
    atomic_write_json(path, batch.to_dict())


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.checkpoints, exist_ok=True)

    print(f"[Runner] Watching state: {args.state_json}")
    print(f"[Runner] Writing commands: {args.cmd_json}")
    print(f"[Runner] Media dir: {args.media_dir}")
    print(f"[Runner] Checkpoints: {args.checkpoints}")

    high = HighLevelAgent(
        weights_path  = os.path.join(args.checkpoints, "high_level"),
        step_interval = 10.0,
    )
    low = LowLevelAgent(
        media_dir     = args.media_dir,
        weights_path  = os.path.join(args.checkpoints, "low_level"),
        step_interval = 0.5,
    )

    poll_interval = 1.0 / args.poll_hz
    last_frame    = -1
    state         = EngineState()   # last known state

    print("[Runner] Starting loop. Ctrl-C to stop.")
    try:
        while True:
            loop_start = time.time()

            # ── 1. Read state ──────────────────────────────────────────────
            new_state = read_state(args.state_json)
            if new_state is not None and new_state.frame_number != last_frame:
                state      = new_state
                last_frame = state.frame_number
                high.observe(state)   # keep history buffer fresh

            # ── 2. High-level step (slow) ──────────────────────────────────
            if high.should_step():
                goal = high.step(state)
                low.set_goal(goal)
                print(f"[Runner] HighLevel goal → {goal}")

            # ── 3. Low-level step (fast) ───────────────────────────────────
            if low.should_step():
                batch = low.step(state)
                if batch.commands:
                    write_commands(args.cmd_json, batch)
                    print(f"[Runner] LowLevel → {len(batch.commands)} command(s): "
                          + ", ".join(c.get("type","?") for c in batch.commands))

            # ── 4. Rate-limit ──────────────────────────────────────────────
            elapsed = time.time() - loop_start
            sleep_t = poll_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Runner] Interrupted. Saving weights...")
        high.policy.save(os.path.join(args.checkpoints, "high_level"))
        low.policy.save(os.path.join(args.checkpoints, "low_level"))
        print(f"[Runner] Done. HighLevel total reward: {high.total_reward:.2f}  "
              f"LowLevel total reward: {low.total_reward:.2f}")


if __name__ == "__main__":
    main()
