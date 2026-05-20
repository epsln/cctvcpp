"""
high_level_agent.py  (v2 — PyTorch + audio conditioning)
"""
from __future__ import annotations
import time, os
import numpy as np

from shared_types   import EngineState
from features       import HistoryBuffer, HIGH_LEVEL_DIM, BASE_DIM
from audio_pipeline import AudioFeatures, AUDIO_FEAT_DIM
from rewards        import FlickerTracker, compute_high_level_reward
from networks       import PPOTrainer

HOLD=0; CALM=1; MEDIUM=2; HIGH_ENERGY=3; TRANSITION=4; GLITCH_MODE=5; AMBIENT_MODE=6
N_ACTIONS=7

GOAL_DEFS={
    HOLD:        {"target_intensity":None,  "preferred_shaders":None,                              "transition":False},
    CALM:        {"target_intensity":0.1,   "preferred_shaders":["color_grade"],                   "transition":False},
    MEDIUM:      {"target_intensity":0.4,   "preferred_shaders":["color_grade","bloom"],            "transition":False},
    HIGH_ENERGY: {"target_intensity":0.9,   "preferred_shaders":["glitch","chromatic","bloom"],     "transition":False},
    TRANSITION:  {"target_intensity":0.5,   "preferred_shaders":["color_grade"],                   "transition":True},
    GLITCH_MODE: {"target_intensity":0.8,   "preferred_shaders":["glitch","chromatic","feedback"],  "transition":False},
    AMBIENT_MODE:{"target_intensity":0.3,   "preferred_shaders":["kaleidoscope","feedback","color_grade"],"transition":False},
}
GOAL_NAMES={0:"HOLD",1:"CALM",2:"MEDIUM",3:"HIGH",4:"TRANSIT",5:"GLITCH",6:"AMBIENT"}

class HighLevelAgent:
    def __init__(self, weights_path="checkpoints/high_level",
                 step_interval=10.0, device="cpu", rollout_steps=32):
        self.weights_path=weights_path; self.step_interval=step_interval
        self._last_step_t=0.0; self._current_goal=GOAL_DEFS[CALM].copy()
        self._current_action=CALM; self.history=HistoryBuffer()

        # High-level visual obs = history window; audio obs = same dim as low
        self.trainer=PPOTrainer(visual_dim=HIGH_LEVEL_DIM, audio_dim=AUDIO_FEAT_DIM,
            n_actions=N_ACTIONS, rollout_steps=rollout_steps,
            ppo_epochs=4, lr=1e-4, device=device, name="high_level")
        self.trainer.load(weights_path)
        self.flicker=FlickerTracker(alpha=0.05)

        self._prev_vis=self.history.as_vector()
        self._prev_aud=np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
        self._prev_action=CALM; self._prev_state=EngineState()
        self._prev_lp=0.0; self._prev_value=0.0
        self.step_n=0; self.total_reward=0.0

    def should_step(self): return (time.time()-self._last_step_t)>=self.step_interval
    def current_goal(self): return self._current_goal.copy()

    def observe(self, state, audio_features=None):
        self.history.push(state)

    @staticmethod
    def _audio_to_array(af):
        if af is None:
            return np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
        scalars=np.array([af.rms_energy,af.spectral_centroid,af.spectral_bandwidth,
            af.spectral_rolloff,af.onset_strength,af.beat_phase,
            af.arousal,af.dominance,af.valence],dtype=np.float32)
        return np.concatenate([af.mel_spectrum,scalars])

    def step(self, state, audio_features=None):
        vis_obs=self.history.as_vector()
        aud_obs=self._audio_to_array(audio_features)

        if self.step_n>0:
            reward, comp=compute_high_level_reward(
                state=state, prev_state=self._prev_state, flicker=self.flicker,
                did_goal_change=(self._prev_action!=self._current_action))
            self.total_reward+=reward
            self.trainer.store(self._prev_vis, self._prev_aud,
                self._prev_action, self._prev_lp, reward, self._prev_value, done=False)
            print(f"[HL {self.step_n:3d}] goal={GOAL_NAMES.get(self._current_action,'?')} "
                  f"r={reward:+.3f} src={comp['source']:+.2f} flick={comp['flicker']:+.2f}")
            if self.trainer.should_update():
                info=self.trainer.update(vis_obs, aud_obs)
                print(f"[HL PPO] upd={info['update']} π={info['policy_loss']:.4f} "
                      f"V={info['value_loss']:.4f}")
                self.trainer.save(self.weights_path)

        action, log_p, value=self.trainer.act(vis_obs, aud_obs)

        # Audio-conditioned override: map high arousal → high energy goal
        if audio_features is not None:
            if audio_features.arousal>0.75 and action==CALM:
                action=MEDIUM
            if audio_features.arousal>0.85 and audio_features.onset_strength>0.7:
                action=HIGH_ENERGY

        goal=GOAL_DEFS[action].copy()
        if action==HOLD: goal=self._current_goal.copy(); goal["transition"]=False

        self._current_goal=goal; self._current_action=action
        self._prev_vis=vis_obs; self._prev_aud=aud_obs
        self._prev_action=action; self._prev_lp=log_p; self._prev_value=value
        self._prev_state=state; self._last_step_t=time.time(); self.step_n+=1
        return goal
