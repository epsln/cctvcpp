"""
low_level_agent.py
──────────────────
The low-level policy operates at ~2 Hz (every 0.5 s).
It controls *concrete* pipeline operations:
  - Which video source to load (from a media library)
  - Which shader passes to add / remove
  - Uniform fine-tuning on active passes

Action space (discrete):
  0           NOOP
  1-6         Toggle shader[i]  (add if absent, remove if present)
  7           Switch to next video in library
  8           Switch to previous video in library
  9           Increase active shader intensity (uEnergy-driven uniform tweak)
  10          Decrease active shader intensity

Total: 11 actions
"""

from __future__ import annotations
import os, glob, time, json
import numpy as np

from shared_types import EngineState, Command, CommandBatch, atomic_write_json
from features import base_features, KNOWN_SHADERS, SHADER_IDX, BASE_DIM
from rewards import FlickerTracker, compute_low_level_reward
from policy import MLPActorCritic

# ─────────────────────────────────────────────────────────────────────────────

NOOP           = 0
TOGGLE_SHADER  = slice(1, 7)   # actions 1–6 → shader index 0–5
NEXT_VIDEO     = 7
PREV_VIDEO     = 8
INTENSITY_UP   = 9
INTENSITY_DOWN = 10
N_ACTIONS      = 11

# Structural change = anything except NOOP and intensity tweaks
_STRUCTURAL_ACTIONS = set(range(1, 9))


def _is_structural(action: int) -> bool:
    return action in _STRUCTURAL_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────

class LowLevelAgent:
    def __init__(
        self,
        media_dir:    str,
        weights_path: str = "checkpoints/low_level",
        hidden_dim:   int = 64,
        step_interval: float = 0.5,   # seconds between decisions
    ):
        self.media_dir     = media_dir
        self.weights_path  = weights_path
        self.step_interval = step_interval
        self._last_step_t  = 0.0
        self._intensity    = 0.5   # tracks our current uniform intensity setting

        # Media library
        self.library = self._scan_library()
        self.lib_idx = 0

        # Policy
        self.policy = MLPActorCritic(
            input_dim  = BASE_DIM,
            n_actions  = N_ACTIONS,
            hidden_dim = hidden_dim,
            name       = "low_level",
        )
        self.policy.load(weights_path)

        # Reward infrastructure
        self.flicker = FlickerTracker(alpha=0.2)

        # Episode memory for update
        self._prev_obs:    np.ndarray  = base_features(EngineState())
        self._prev_action: int         = NOOP
        self._prev_log_p:  float       = 0.0
        self._prev_value:  float       = 0.0
        self._prev_state:  EngineState = EngineState()

        self.total_reward = 0.0
        self.step_n       = 0

    # ── Library ─────────────────────────────────

    def _scan_library(self) -> list[str]:
        exts = ["*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.jpg", "*.png"]
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(self.media_dir, ext))
        files.sort()
        if not files:
            print(f"[LowLevel] WARNING: no media found in {self.media_dir}")
        return files

    # ── Goal setter (called by high-level) ──────

    def set_goal(self, goal: dict) -> None:
        """
        High-level passes a goal dict, e.g.:
          {"target_intensity": 0.8, "preferred_shaders": ["bloom", "glitch"]}
        Low-level uses this to bias its behaviour via observation augmentation.
        (For now stored as context; future: concatenated to obs vector.)
        """
        self._goal = goal

    # ── Decision step ────────────────────────────

    def should_step(self) -> bool:
        return (time.time() - self._last_step_t) >= self.step_interval

    def step(self, state: EngineState) -> CommandBatch:
        """
        Observe state, pick action, build command batch.
        Also performs the RL update using the *previous* transition.
        """
        obs = base_features(state)

        # ── RL update on previous transition ──────
        if self.step_n > 0:
            reward, components = compute_low_level_reward(
                state       = state,
                prev_state  = self._prev_state,
                flicker     = self.flicker,
                did_structural_change = _is_structural(self._prev_action),
            )
            self.total_reward += reward
            train_info = self.policy.update(
                obs      = self._prev_obs,
                action   = self._prev_action,
                reward   = reward,
                next_obs = obs,
                done     = False,
            )
            if self.step_n % 20 == 0:
                print(f"[LowLevel step {self.step_n}] "
                      f"reward={reward:.3f} "
                      f"(src={components['source']:.2f} "
                      f"flick={components['flicker']:.2f} "
                      f"crowd={components['crowd']:.2f}) "
                      f"adv={train_info['advantage']:.3f} "
                      f"flicker_rate={self.flicker.action_rate:.3f}")

            if self.step_n % 200 == 0:
                self.policy.save(self.weights_path)

        # ── Choose action ─────────────────────────
        action, log_p, value = self.policy.act(obs)

        # ── Build commands ────────────────────────
        cmds = self._action_to_commands(action, state)
        batch = CommandBatch(
            commands    = [c.to_dict() for c in cmds],
            agent_level = "low",
        )

        # ── Store for next update ─────────────────
        self._prev_obs    = obs
        self._prev_action = action
        self._prev_log_p  = log_p
        self._prev_value  = value
        self._prev_state  = state
        self._last_step_t = time.time()
        self.step_n      += 1

        return batch

    # ── Action → commands translation ────────────

    def _action_to_commands(self, action: int, state: EngineState) -> list[Command]:
        cmds: list[Command] = []

        if action == NOOP:
            pass

        elif 1 <= action <= 6:
            # Toggle shader (action 1 → KNOWN_SHADERS[0], etc.)
            shader_idx  = action - 1
            shader_name = KNOWN_SHADERS[shader_idx]
            pass_id     = f"rl_{shader_name}"

            # Check if this shader is currently in the pipeline
            active_shaders = {
                (p.get("shader") if isinstance(p, dict) else p.shader)
                for p in state.passes
                if (p.get("enabled", True) if isinstance(p, dict) else p.enabled)
            }

            if shader_name in active_shaders:
                cmds.append(Command(type="REMOVE_PASS", pass_id=pass_id))
            else:
                cmds.append(Command(
                    type="ADD_PASS",
                    pass_id=pass_id,
                    shader_name=shader_name,
                    position=-1,
                ))
                # Apply sensible defaults
                cmds += self._default_uniforms(pass_id, shader_name)

        elif action == NEXT_VIDEO:
            if self.library:
                self.lib_idx = (self.lib_idx + 1) % len(self.library)
                cmds.append(Command(
                    type="SET_SOURCE",
                    source_path=self.library[self.lib_idx],
                ))

        elif action == PREV_VIDEO:
            if self.library:
                self.lib_idx = (self.lib_idx - 1) % len(self.library)
                cmds.append(Command(
                    type="SET_SOURCE",
                    source_path=self.library[self.lib_idx],
                ))

        elif action == INTENSITY_UP:
            self._intensity = min(1.0, self._intensity + 0.15)
            cmds += self._intensity_commands(state, self._intensity)

        elif action == INTENSITY_DOWN:
            self._intensity = max(0.0, self._intensity - 0.15)
            cmds += self._intensity_commands(state, self._intensity)

        return cmds

    def _default_uniforms(self, pass_id: str, shader: str) -> list[Command]:
        defaults = {
            "chromatic":    [("uStrength", 0.008), ("uBarrel", 0.1)],
            "glitch":       [("uAmount",   0.04),  ("uSpeed",  4.0)],
            "color_grade":  [("uSaturation", 1.3), ("uContrast", 1.1), ("uHueShift", 5.0)],
            "bloom":        [("uThreshold", 0.55), ("uIntensity", 1.2), ("uRadius", 6)],
            "kaleidoscope": [("uSegments", 6.0),   ("uRotation", 0.2)],
            "feedback":     [("uDecay",    0.85),  ("uZoom", 0.995),   ("uSpin", 0.3)],
        }
        cmds = []
        for uname, uval in defaults.get(shader, []):
            utype = "int" if isinstance(uval, int) else "float"
            cmds.append(Command(
                type="SET_UNIFORM",
                pass_id=pass_id,
                uniform_name=uname,
                uniform_type=utype,
                uniform_value=uval,
            ))
        return cmds

    def _intensity_commands(self, state: EngineState, intensity: float) -> list[Command]:
        """Broadcast an intensity-driven uniform to all active RL passes."""
        cmds = []
        for p in state.passes:
            pid    = p.get("id")     if isinstance(p, dict) else p.id
            shader = p.get("shader") if isinstance(p, dict) else p.shader
            if not pid or not pid.startswith("rl_"):
                continue
            if shader == "glitch":
                cmds.append(Command(type="SET_UNIFORM", pass_id=pid,
                                     uniform_name="uAmount", uniform_type="float",
                                     uniform_value=intensity * 0.08))
            elif shader == "bloom":
                cmds.append(Command(type="SET_UNIFORM", pass_id=pid,
                                     uniform_name="uIntensity", uniform_type="float",
                                     uniform_value=0.5 + intensity * 2.0))
            elif shader == "chromatic":
                cmds.append(Command(type="SET_UNIFORM", pass_id=pid,
                                     uniform_name="uStrength", uniform_type="float",
                                     uniform_value=intensity * 0.025))
        return cmds
