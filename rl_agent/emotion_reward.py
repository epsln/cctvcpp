"""
emotion_reward.py
─────────────────
Computes the reward signal that steers the RL agent toward producing visuals
that emotionally correspond to the audio.

Reward components
─────────────────
R_emotion   — L2 distance between audio and visual emotion in the
              (arousal, valence) plane. Primary learning signal.

R_beat_sync — reward for structural changes (shader add/remove, source switch)
              that happen close to an audio onset. Makes cuts feel musical.

R_energy    — visual entropy should track audio RMS energy.
              Prevents the agent from learning to ignore energy changes.

R_stability — small bonus for not changing anything (penalises jitter near
              correct states). Complements the existing flicker penalty.

All components return values in [-1, 1] before weighting.
The weights are chosen so R_emotion dominates (~60%), with the others shaping
the trajectory but not overpowering it.
"""

from __future__ import annotations
import math
import time
import numpy as np
from dataclasses import dataclass

from audio_pipeline import AudioFeatures


# ─────────────────────────────────────────────────────────────────────────────
#  Reward weights
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmotionRewardWeights:
    emotion:   float = 3.0   # audio ↔ visual emotion alignment (dominant)
    beat_sync: float = 1.5   # structural changes on beats
    energy:    float = 1.0   # visual entropy vs audio RMS
    stability: float = 0.5   # reward for staying still when already correct


DEFAULT_EMOTION_WEIGHTS = EmotionRewardWeights()


# ─────────────────────────────────────────────────────────────────────────────
#  Beat sync tracker
# ─────────────────────────────────────────────────────────────────────────────

class BeatSyncTracker:
    """
    Tracks whether structural actions land close to audio onsets.
    We maintain a rolling window of recent onset times and check
    whether each structural action was within `tolerance` seconds
    of the nearest onset.
    """

    def __init__(self, tolerance_sec: float = 0.15, window_sec: float = 4.0):
        self.tolerance   = tolerance_sec
        self.window      = window_sec
        self._onsets: list[float] = []   # wall-clock times of detected onsets
        self._last_onset_strength = 0.0
        self._onset_threshold = 0.6

    def update_audio(self, features: AudioFeatures, wall_time: float) -> None:
        """Call every audio feature update."""
        # Edge detect: onset_strength crosses threshold upward
        if (features.onset_strength > self._onset_threshold and
                self._last_onset_strength <= self._onset_threshold):
            self._onsets.append(wall_time)
            # Prune old
            cutoff = wall_time - self.window
            self._onsets = [t for t in self._onsets if t > cutoff]
        self._last_onset_strength = features.onset_strength

    def score_action(self, action_wall_time: float) -> float:
        """
        Returns 1.0 if action was within tolerance of a recent onset,
        0.0 otherwise.  Decays as action moves away from nearest onset.
        """
        if not self._onsets:
            return 0.0
        nearest = min(abs(action_wall_time - t) for t in self._onsets)
        if nearest <= self.tolerance:
            # Cosine taper: 1.0 at 0 offset, 0.0 at tolerance
            return float(math.cos((nearest / self.tolerance) * math.pi / 2.0))
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Visual entropy estimator
# ─────────────────────────────────────────────────────────────────────────────

def visual_entropy(pil_image) -> float:
    """
    Pixel intensity histogram entropy — proxy for visual complexity/energy.
    Returns value in [0, 1] (1 = maximum entropy / most complex).
    """
    try:
        import numpy as np
        img  = np.array(pil_image.convert("L")).ravel()   # greyscale
        hist, _ = np.histogram(img, bins=64, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        # Shannon entropy, normalised to [0,1] by log2(64)
        ent = -float(np.sum(hist * np.log2(hist + 1e-10)))
        return float(np.clip(ent / math.log2(64), 0.0, 1.0))
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  EmotionRewardComputer
# ─────────────────────────────────────────────────────────────────────────────

class EmotionRewardComputer:
    """
    Stateful reward computer.  Call update() once per RL step.
    Maintains the beat sync tracker and last-known emotion values.
    """

    def __init__(self, weights: EmotionRewardWeights = DEFAULT_EMOTION_WEIGHTS):
        self.weights    = weights
        self.beat_sync  = BeatSyncTracker()

        self._last_visual_entropy = 0.5

    def update_audio(self, features: AudioFeatures, wall_time: float) -> None:
        """Must be called every audio pipeline tick (before compute)."""
        self.beat_sync.update_audio(features, wall_time)

    def compute(
        self,
        audio_features:   AudioFeatures,
        visual_arousal:   float,
        visual_valence:   float,
        pil_image,
        did_structural_change: bool,
        action_wall_time: float,
    ) -> tuple[float, dict]:
        """
        Returns (total_reward, component_dict).

        audio_features:  latest AudioFeatures from AudioPipeline
        visual_arousal:  from VisualEmotionEstimator [-1, 1]
        visual_valence:  from VisualEmotionEstimator [-1, 1]
        pil_image:       latest captured frame (PIL Image or None)
        did_structural_change: whether the agent changed the pipeline this step
        action_wall_time: wall clock time of the action
        """

        # ── R_emotion ──────────────────────────────────────────────────────
        # Map audio emotion to [-1, 1] (it arrives in [0, 1])
        aud_arousal = (audio_features.arousal  - 0.5) * 2.0
        aud_valence = (audio_features.valence  - 0.5) * 2.0

        dist = math.sqrt(
            (aud_arousal - visual_arousal) ** 2 +
            (aud_valence - visual_valence) ** 2
        )
        # Max possible distance in the [-1,1]^2 square is 2√2 ≈ 2.83
        r_emotion = float(1.0 - dist / (2.0 * math.sqrt(2.0)) * 2.0)
        r_emotion = max(-1.0, min(1.0, r_emotion))

        # ── R_beat_sync ────────────────────────────────────────────────────
        if did_structural_change:
            r_beat = self.beat_sync.score_action(action_wall_time) * 2.0 - 1.0
            # +1 if on beat, -1 if off beat (only charged when action happens)
        else:
            r_beat = 0.0   # no action → no beat sync reward/penalty

        # ── R_energy ───────────────────────────────────────────────────────
        if pil_image is not None:
            vis_ent = visual_entropy(pil_image)
            self._last_visual_entropy = vis_ent
        else:
            vis_ent = self._last_visual_entropy

        # Audio energy in [0, 1]; visual entropy in [0, 1] → align them
        energy_gap = abs(audio_features.rms_energy - vis_ent)
        r_energy = float(1.0 - energy_gap * 2.0)  # 0 gap → +1, 0.5 gap → 0

        # ── R_stability ────────────────────────────────────────────────────
        # Small reward for not changing anything when emotion alignment is good
        if not did_structural_change and r_emotion > 0.5:
            r_stability = 0.3
        else:
            r_stability = 0.0

        components = {
            "emotion":   r_emotion   * self.weights.emotion,
            "beat_sync": r_beat      * self.weights.beat_sync,
            "energy":    r_energy    * self.weights.energy,
            "stability": r_stability * self.weights.stability,
        }
        total = sum(components.values())
        return float(total), components
