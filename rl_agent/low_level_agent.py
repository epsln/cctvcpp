"""
low_level_agent.py  (v2 — PyTorch + audio conditioning)
"""
from __future__ import annotations
import glob, os, time
import numpy as np

from shared_types   import EngineState, Command, CommandBatch
from features       import base_features, KNOWN_SHADERS, BASE_DIM
from audio_pipeline import AudioFeatures, AUDIO_FEAT_DIM
from emotion_reward import EmotionRewardComputer
from rewards        import FlickerTracker, compute_low_level_reward
from networks       import PPOTrainer

NOOP=0; NEXT_VIDEO=7; PREV_VIDEO=8; INTENSITY_UP=9; INTENSITY_DOWN=10; N_ACTIONS=11
_STRUCTURAL = set(range(1, 9))
def _is_structural(a): return a in _STRUCTURAL

class LowLevelAgent:
    def __init__(self, media_dir, weights_path="checkpoints/low_level",
                 step_interval=0.5, device="cpu", rollout_steps=64):
        self.media_dir=media_dir; self.weights_path=weights_path
        self.step_interval=step_interval; self._last_step_t=0.0
        self._intensity=0.5; self._goal={}
        self.library=self._scan_library(); self.lib_idx=0
        self.trainer=PPOTrainer(visual_dim=BASE_DIM, audio_dim=AUDIO_FEAT_DIM,
            n_actions=N_ACTIONS, rollout_steps=rollout_steps, device=device, name="low_level")
        self.trainer.load(weights_path)
        self.flicker=FlickerTracker(alpha=0.2)
        self.emotion_reward=EmotionRewardComputer()
        self._prev_vis=base_features(EngineState())
        self._prev_aud=np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
        self._prev_action=NOOP; self._prev_lp=0.0; self._prev_value=0.0
        self._prev_state=EngineState(); self._prev_action_t=0.0
        self.step_n=0; self.total_reward=0.0

    def _scan_library(self):
        files=[]
        for ext in ["*.mp4","*.mov","*.mkv","*.avi","*.webm","*.jpg","*.png"]:
            files+=glob.glob(os.path.join(self.media_dir, ext))
        files.sort()
        if not files: print(f"[LowLevel] WARNING: no media in {self.media_dir}")
        return files

    def set_goal(self, goal): self._goal=goal
    def should_step(self): return (time.time()-self._last_step_t)>=self.step_interval

    @staticmethod
    def _audio_to_array(af):
        scalars=np.array([af.rms_energy,af.spectral_centroid,af.spectral_bandwidth,
            af.spectral_rolloff,af.onset_strength,af.beat_phase,
            af.arousal,af.dominance,af.valence],dtype=np.float32)
        return np.concatenate([af.mel_spectrum,scalars])

    def step(self, state, audio_features, visual_arousal=0.0, visual_valence=0.0, pil_frame=None):
        vis_obs=base_features(state)
        aud_obs=self._audio_to_array(audio_features)
        now=time.time()

        if self.step_n>0:
            self.emotion_reward.update_audio(audio_features, now)
            emo_r, emo_comp=self.emotion_reward.compute(
                audio_features=audio_features,
                visual_arousal=visual_arousal, visual_valence=visual_valence,
                pil_image=pil_frame, did_structural_change=_is_structural(self._prev_action),
                action_wall_time=self._prev_action_t)
            cont_r, cont_comp=compute_low_level_reward(
                state=state, prev_state=self._prev_state, flicker=self.flicker,
                did_structural_change=_is_structural(self._prev_action))
            reward=emo_r+cont_r
            self.total_reward+=reward
            self.trainer.store(self._prev_vis, self._prev_aud,
                self._prev_action, self._prev_lp, reward, self._prev_value, done=False)
            if self.step_n%10==0:
                print(f"[LL {self.step_n:4d}] r={reward:+.3f} "
                      f"emo={emo_comp['emotion']:+.2f} beat={emo_comp['beat_sync']:+.2f} "
                      f"A={audio_features.arousal:.2f} V={audio_features.valence:.2f} "
                      f"vA={visual_arousal:.2f} vV={visual_valence:.2f}")
            if self.trainer.should_update():
                info=self.trainer.update(vis_obs, aud_obs)
                print(f"[LL PPO] upd={info['update']} π={info['policy_loss']:.4f} "
                      f"V={info['value_loss']:.4f} H={info['entropy']:.4f}")
                self.trainer.save(self.weights_path)

        action, log_p, value=self.trainer.act(vis_obs, aud_obs)
        cmds=self._action_to_commands(action, state)
        self._prev_vis=vis_obs; self._prev_aud=aud_obs
        self._prev_action=action; self._prev_lp=log_p; self._prev_value=value
        self._prev_state=state; self._prev_action_t=now
        self._last_step_t=now; self.step_n+=1
        return CommandBatch(commands=[c.to_dict() for c in cmds], agent_level="low")

    def _action_to_commands(self, action, state):
        cmds=[]
        if action==NOOP: pass
        elif 1<=action<=6:
            shader_name=KNOWN_SHADERS[action-1]; pass_id=f"rl_{shader_name}"
            active={(p.get("shader") if isinstance(p,dict) else p.shader)
                    for p in state.passes
                    if (p.get("enabled",True) if isinstance(p,dict) else p.enabled)}
            if shader_name in active:
                cmds.append(Command(type="REMOVE_PASS", pass_id=pass_id))
            else:
                cmds.append(Command(type="ADD_PASS", pass_id=pass_id,
                    shader_name=shader_name, position=-1))
                cmds+=self._default_uniforms(pass_id, shader_name)
        elif action==NEXT_VIDEO:
            if self.library:
                self.lib_idx=(self.lib_idx+1)%len(self.library)
                cmds.append(Command(type="SET_SOURCE",source_path=self.library[self.lib_idx]))
        elif action==PREV_VIDEO:
            if self.library:
                self.lib_idx=(self.lib_idx-1)%len(self.library)
                cmds.append(Command(type="SET_SOURCE",source_path=self.library[self.lib_idx]))
        elif action==INTENSITY_UP:
            self._intensity=min(1.0,self._intensity+0.15)
            cmds+=self._intensity_commands(state,self._intensity)
        elif action==INTENSITY_DOWN:
            self._intensity=max(0.0,self._intensity-0.15)
            cmds+=self._intensity_commands(state,self._intensity)
        return cmds

    def _default_uniforms(self, pass_id, shader):
        defaults={"chromatic":[("uStrength",0.008),("uBarrel",0.1)],
            "glitch":[("uAmount",0.04),("uSpeed",4.0)],
            "color_grade":[("uSaturation",1.3),("uContrast",1.1),("uHueShift",5.0)],
            "bloom":[("uThreshold",0.55),("uIntensity",1.2),("uRadius",6)],
            "kaleidoscope":[("uSegments",6.0),("uRotation",0.2)],
            "feedback":[("uDecay",0.85),("uZoom",0.995),("uSpin",0.3)]}
        return [Command(type="SET_UNIFORM", pass_id=pass_id,
            uniform_name=n, uniform_type="int" if isinstance(v,int) else "float",
            uniform_value=v) for n,v in defaults.get(shader,[])]

    def _intensity_commands(self, state, intensity):
        cmds=[]
        for p in state.passes:
            pid=p.get("id") if isinstance(p,dict) else p.id
            shader=p.get("shader") if isinstance(p,dict) else p.shader
            if not pid or not pid.startswith("rl_"): continue
            for s,(un,uv) in {"glitch":("uAmount",intensity*0.08),
                               "bloom":("uIntensity",0.5+intensity*2.0),
                               "chromatic":("uStrength",intensity*0.025)}.items():
                if shader==s:
                    cmds.append(Command(type="SET_UNIFORM",pass_id=pid,
                        uniform_name=un,uniform_type="float",uniform_value=uv))
        return cmds
