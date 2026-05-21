"""
low_level_agent.py  (v3 — registry-driven actions + MLflow)
"""
from __future__ import annotations
import glob, os, time
import numpy as np

from shared_types    import EngineState, Command, CommandBatch
from features        import base_features, BASE_DIM
from audio_pipeline  import AudioFeatures, AUDIO_FEAT_DIM
from emotion_reward  import EmotionRewardComputer
from rewards         import FlickerTracker, compute_low_level_reward
from networks        import PPOTrainer
from metrics         import Metrics
from shader_registry import (
    ACTION_SPACE, N_ACTIONS, ActionSpec,
    NOOP_IDX, TOGGLE_MODIFIER_START, TOGGLE_MODIFIER_END,
    TOGGLE_FEEDBACK_IDX, NEXT_VIDEO_IDX, PREV_VIDEO_IDX,
    INTENSITY_UP_IDX, INTENSITY_DOWN_IDX,
    is_structural, BY_NAME, FEEDBACK_SOURCE_NAMES,
    validate_pipeline, version_hash,
)


class LowLevelAgent:
    def __init__(
        self,
        media_dir:     str,
        weights_path:  str     = "checkpoints/low_level",
        step_interval: float   = 0.5,
        device:        str     = "cpu",
        rollout_steps: int     = 64,
        metrics:       Metrics = None,
    ):
        self.media_dir     = media_dir
        self.weights_path  = weights_path
        self.step_interval = step_interval
        self._last_step_t  = 0.0
        self._intensity    = 0.5
        self._goal: dict   = {}
        self.metrics       = metrics or Metrics(disabled=True)

        self.library = self._scan_library()
        self.lib_idx = 0

        # Weights path includes a registry hash so stale checkpoints are detected
        versioned_path = f"{weights_path}_{version_hash()}"
        self.trainer = PPOTrainer(
            visual_dim    = BASE_DIM,
            audio_dim     = AUDIO_FEAT_DIM,
            n_actions     = N_ACTIONS,
            rollout_steps = rollout_steps,
            device        = device,
            name          = "low_level",
        )
        if not self.trainer.load(versioned_path):
            print(f"[LowLevel] No checkpoint at {versioned_path}, starting fresh.")
        self.weights_path_versioned = versioned_path

        self.flicker        = FlickerTracker(alpha=0.2)
        self.emotion_reward = EmotionRewardComputer()

        # Per-step memory
        self._prev_vis    = base_features(EngineState())
        self._prev_aud    = np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
        self._prev_action = NOOP_IDX
        self._prev_lp     = 0.0
        self._prev_value  = 0.0
        self._prev_state  = EngineState()
        self._prev_action_t = 0.0
        self.step_n       = 0
        self.total_reward = 0.0

        # Log hyperparams once
        self.metrics.log_params({
            "ll_rollout_steps":  rollout_steps,
            "ll_step_interval":  step_interval,
            "ll_n_actions":      N_ACTIONS,
            "ll_base_dim":       BASE_DIM,
            "registry_hash":     version_hash(),
        })

    # ── Library ──────────────────────────────────────────────────────────────

    def _scan_library(self) -> list[str]:
        files = []
        for ext in ["*.mp4","*.mov","*.mkv","*.avi","*.webm","*.jpg","*.png"]:
            files += glob.glob(os.path.join(self.media_dir, ext))
        files.sort()
        if not files:
            print(f"[LowLevel] WARNING: no media in {self.media_dir}")
        return files

    def set_goal(self, goal: dict) -> None:
        self._goal = goal

    def should_step(self) -> bool:
        return (time.time() - self._last_step_t) >= self.step_interval

    # ── Audio obs ────────────────────────────────────────────────────────────

    @staticmethod
    def _audio_to_array(af: AudioFeatures) -> np.ndarray:
        scalars = np.array([
            af.rms_energy, af.spectral_centroid, af.spectral_bandwidth,
            af.spectral_rolloff, af.onset_strength, af.beat_phase,
            af.arousal, af.dominance, af.valence,
        ], dtype=np.float32)
        return np.concatenate([af.mel_spectrum, scalars])

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(
        self,
        state:          EngineState,
        audio_features: AudioFeatures,
        visual_arousal: float = 0.0,
        visual_valence: float = 0.0,
        pil_frame       = None,
    ) -> CommandBatch:

        vis_obs = base_features(state)
        aud_obs = self._audio_to_array(audio_features)
        now     = time.time()

        # ── Reward + PPO store for previous step ─────────────────────────────
        if self.step_n > 0:
            self.emotion_reward.update_audio(audio_features, now)
            emo_r, emo_comp = self.emotion_reward.compute(
                audio_features        = audio_features,
                visual_arousal        = visual_arousal,
                visual_valence        = visual_valence,
                pil_image             = pil_frame,
                did_structural_change = is_structural(self._prev_action),
                action_wall_time      = self._prev_action_t,
            )
            cont_r, cont_comp = compute_low_level_reward(
                state                 = state,
                prev_state            = self._prev_state,
                flicker               = self.flicker,
                did_structural_change = is_structural(self._prev_action),
            )
            reward = emo_r + cont_r
            self.total_reward += reward

            # Merge component dicts for logging
            all_comp = {**{f"emo_{k}": v for k, v in emo_comp.items()},
                        **{f"cont_{k}": v for k, v in cont_comp.items()}}
            self.metrics.log_low_reward(all_comp, reward)
            self.metrics.log_visual(visual_arousal, visual_valence)
            self.metrics.log_emotion_alignment(
                audio_arousal  = (audio_features.arousal - 0.5) * 2.0,
                audio_valence  = (audio_features.valence - 0.5) * 2.0,
                visual_arousal = visual_arousal,
                visual_valence = visual_valence,
            )
            # Log per-shader active state
            active = [p.get("shader","") if isinstance(p,dict) else p.shader
                      for p in state.passes
                      if (p.get("enabled",True) if isinstance(p,dict) else p.enabled)]
            self.metrics.log_shader_state(active)

            # Pipeline constraint check — log violations
            valid, reason = validate_pipeline(active)
            if not valid:
                self.metrics._log("pipeline/constraint_violation", 1.0, self.step_n)
                print(f"[LowLevel] Pipeline constraint violation: {reason}")

            self.trainer.store(
                self._prev_vis, self._prev_aud,
                self._prev_action, self._prev_lp,
                reward, self._prev_value, done=False,
            )

            if self.step_n % 10 == 0:
                print(f"[LL {self.step_n:4d}] r={reward:+.3f} "
                      f"emo={emo_comp['emotion']:+.2f} "
                      f"beat={emo_comp['beat_sync']:+.2f} "
                      f"A={audio_features.arousal:.2f} V={audio_features.valence:.2f} "
                      f"vA={visual_arousal:.2f} vV={visual_valence:.2f}")

            if self.trainer.should_update():
                info = self.trainer.update(vis_obs, aud_obs)
                # Augment info with return stats for logging
                info["returns_mean"] = float(
                    self.trainer.buffer.rewards[:self.trainer.buffer.ptr].mean()
                    if self.trainer.buffer.ptr > 0 else 0.0
                )
                self.metrics.log_low_ppo(info)
                self.metrics.log_checkpoint(self.weights_path_versioned + ".pt")
                print(f"[LL PPO] upd={info['update']} "
                      f"π={info['policy_loss']:.4f} "
                      f"V={info['value_loss']:.4f} "
                      f"H={info['entropy']:.4f}")
                self.trainer.save(self.weights_path_versioned)

        # ── Choose action ─────────────────────────────────────────────────────
        action, log_p, value = self.trainer.act(vis_obs, aud_obs)

        # Constrain: if feedback is active and no modifier is active, block
        # adding feedback again (prevent double-feedback)
        action = self._constrain_action(action, state)

        action_label = ACTION_SPACE[action].label
        self.metrics.log_action("low", action_label)
        self.metrics.tick_low()

        cmds = self._action_to_commands(action, state)

        # Store for next step
        self._prev_vis      = vis_obs
        self._prev_aud      = aud_obs
        self._prev_action   = action
        self._prev_lp       = log_p
        self._prev_value    = value
        self._prev_state    = state
        self._prev_action_t = now
        self._last_step_t   = now
        self.step_n        += 1

        return CommandBatch(commands=[c.to_dict() for c in cmds], agent_level="low")

    # ── Constraint enforcement ────────────────────────────────────────────────

    def _constrain_action(self, action: int, state: EngineState) -> int:
        """
        Enforce pipeline constraints before executing.
        Converts illegal actions to NOOP rather than crashing.
        """
        spec = ACTION_SPACE[action]

        if spec.kind == "toggle_feedback":
            # Only allow feedback if at least one modifier is already active
            active = [p.get("shader","") if isinstance(p,dict) else p.shader
                      for p in state.passes
                      if (p.get("enabled",True) if isinstance(p,dict) else p.enabled)]
            from shader_registry import MODIFIER_NAMES
            has_modifier = any(s in MODIFIER_NAMES for s in active)
            if not has_modifier:
                return NOOP_IDX  # silently block

        return action

    # ── Action → commands ─────────────────────────────────────────────────────

    def _action_to_commands(self, action: int, state: EngineState) -> list[Command]:
        cmds: list[Command] = []
        spec = ACTION_SPACE[action]

        if spec.kind == "noop":
            pass

        elif spec.kind == "toggle_modifier":
            shader_name = spec.shader_name
            pass_id     = f"rl_{shader_name}"
            active = {
                (p.get("shader","") if isinstance(p,dict) else p.shader)
                for p in state.passes
                if (p.get("enabled",True) if isinstance(p,dict) else p.enabled)
            }
            if shader_name in active:
                cmds.append(Command(type="REMOVE_PASS", pass_id=pass_id))
            else:
                # Determine insertion position from registry spec
                shader_spec = BY_NAME[shader_name]
                pos = 0 if shader_spec.preferred_position == "first" else -1
                cmds.append(Command(type="ADD_PASS", pass_id=pass_id,
                                    shader_name=shader_name, position=pos))
                cmds += self._default_uniforms(pass_id, shader_name)

        elif spec.kind == "toggle_feedback":
            # Toggle the first (and usually only) feedback source
            if FEEDBACK_SOURCE_NAMES:
                fb_name = FEEDBACK_SOURCE_NAMES[0]
                pass_id = f"rl_{fb_name}"
                active = {
                    (p.get("shader","") if isinstance(p,dict) else p.shader)
                    for p in state.passes
                    if (p.get("enabled",True) if isinstance(p,dict) else p.enabled)
                }
                if fb_name in active:
                    cmds.append(Command(type="REMOVE_PASS", pass_id=pass_id))
                else:
                    # Always insert feedback last
                    cmds.append(Command(type="ADD_PASS", pass_id=pass_id,
                                        shader_name=fb_name, position=-1))
                    cmds += self._default_uniforms(pass_id, fb_name)

        elif spec.kind == "next_video":
            if self.library:
                self.lib_idx = (self.lib_idx + 1) % len(self.library)
                cmds.append(Command(type="SET_SOURCE",
                                    source_path=self.library[self.lib_idx]))

        elif spec.kind == "prev_video":
            if self.library:
                self.lib_idx = (self.lib_idx - 1) % len(self.library)
                cmds.append(Command(type="SET_SOURCE",
                                    source_path=self.library[self.lib_idx]))

        elif spec.kind == "intensity_up":
            self._intensity = min(1.0, self._intensity + 0.15)
            cmds += self._intensity_commands(state, self._intensity)

        elif spec.kind == "intensity_down":
            self._intensity = max(0.0, self._intensity - 0.15)
            cmds += self._intensity_commands(state, self._intensity)

        return cmds

    def _default_uniforms(self, pass_id: str, shader_name: str) -> list[Command]:
        spec = BY_NAME.get(shader_name)
        if spec is None:
            return []
        return [
            Command(
                type          = "SET_UNIFORM",
                pass_id       = pass_id,
                uniform_name  = u.name,
                uniform_type  = u.type,
                uniform_value = u.default,
            )
            for u in spec.uniforms
        ]

    def _intensity_commands(self, state: EngineState, intensity: float) -> list[Command]:
        """
        Scale the 'primary intensity' uniform of each active RL pass.
        Which uniform is the 'intensity knob' is read from the registry:
        we use the first uniform with max_val > 0.5 as the target.
        """
        cmds = []
        for p in state.passes:
            pid    = p.get("id","")    if isinstance(p,dict) else getattr(p,"id","")
            shader = p.get("shader","") if isinstance(p,dict) else getattr(p,"shader","")
            if not pid.startswith("rl_"):
                continue
            spec = BY_NAME.get(shader)
            if spec is None:
                continue
            # Find the primary intensity uniform (first float with range > 0.5)
            for u in spec.uniforms:
                if u.type == "float" and (u.max_val - u.min_val) > 0.5:
                    val = u.min_val + intensity * (u.max_val - u.min_val)
                    cmds.append(Command(
                        type="SET_UNIFORM", pass_id=pid,
                        uniform_name=u.name, uniform_type="float",
                        uniform_value=float(val),
                    ))
                    break   # only the first intensity knob per shader
        return cmds
