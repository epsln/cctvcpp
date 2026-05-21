"""
audio_pipeline.py
─────────────────
Processes a streaming audio file (DJ set) and emits, per analysis window:
  - FFT magnitude spectrum (normalised, log-scaled)
  - Onset envelope + estimated BPM
  - Energy RMS
  - Spectral centroid / bandwidth / rolloff
  - Emotion embedding: [arousal, dominance, valence] via audeering wav2vec2

All operations are CPU-friendly; the emotion model runs every ~2 s (configurable).
Everything else runs every ~0.1 s to keep the RL step well-fed.

Usage:
  pipeline = AudioPipeline("djset.mp3")
  pipeline.start()
  # in your loop:
  features = pipeline.get_latest()   # AudioFeatures, non-blocking
  pipeline.stop()
"""

from __future__ import annotations
import threading
import time
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa

# ─────────────────────────────────────────────────────────────────────────────
#  Output feature bundle
# ─────────────────────────────────────────────────────────────────────────────

FFT_BINS = 128   # mel bins exposed to the RL agent

@dataclass
class AudioFeatures:
    # Raw spectrum (log-mel, normalised), shape (FFT_BINS,)
    mel_spectrum:     np.ndarray = field(default_factory=lambda: np.zeros(FFT_BINS, dtype=np.float32))

    # Scalar signal features
    rms_energy:       float = 0.0   # 0-1
    spectral_centroid: float = 0.0  # normalised 0-1
    spectral_bandwidth: float = 0.0
    spectral_rolloff:   float = 0.0
    onset_strength:     float = 0.0  # 0-1, high on beat onsets
    bpm_estimate:       float = 120.0

    # Beat phase: fraction [0,1] within the current beat
    beat_phase:         float = 0.0

    # Emotion (updated every ~2 s, held between updates)
    arousal:    float = 0.5   # 0=calm,    1=excited
    dominance:  float = 0.5   # 0=submissive, 1=dominant
    valence:    float = 0.5   # 0=negative, 1=positive

    # Playback position
    pos_sec:    float = 0.0

    # Flag: True on the first frame after an emotion update
    emotion_updated: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion model wrapper (audeering wav2vec2)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionModel:
    """
    Wraps audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.
    Outputs [arousal, dominance, valence] in [0, 1].
    Lazy-loaded on first call to avoid startup delay.
    Falls back to librosa-based heuristic if model unavailable.
    """
    MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._processor = None
        self._model = None
        self._loaded = False
        self._failed = False

    def _load(self):
        if self._loaded or self._failed:
            return
        try:
            from transformers import Wav2Vec2Processor
            from transformers import Wav2Vec2PreTrainedModel
            import torch.nn as nn

            # The audeering model uses a custom head on top of wav2vec2
            # We load it via the generic AutoModel path
            from transformers import AutoProcessor, AutoModelForAudioClassification

            # Try the specific audeering class first
            try:
                from transformers import Wav2Vec2Processor
                self._processor = Wav2Vec2Processor.from_pretrained(self.MODEL_ID)

                # Load model using the audeering custom class
                from transformers import AutoConfig
                import torch.nn as nn

                class RegressionHead(nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                        self.dropout = nn.Dropout(config.final_dropout)
                        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
                    def forward(self, features):
                        x = features
                        x = self.dropout(x)
                        x = self.dense(x)
                        x = torch.tanh(x)
                        x = self.dropout(x)
                        x = self.out_proj(x)
                        return x

                from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

                class EmotionWav2Vec2(Wav2Vec2PreTrainedModel):
                    def __init__(self, config):
                        super().__init__(config)
                        self.wav2vec2 = Wav2Vec2Model(config)
                        self.classifier = RegressionHead(config)
                        self.init_weights()
                    def forward(self, input_values):
                        outputs = self.wav2vec2(input_values)
                        hidden = outputs.last_hidden_state.mean(dim=1)
                        logits = self.classifier(hidden)
                        return torch.sigmoid(logits)   # [0,1] per dimension

                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.MODEL_ID)
                self._model = EmotionWav2Vec2.from_pretrained(self.MODEL_ID, config=config)
                self._model.eval().to(self.device)
                print("[EmotionModel] Loaded audeering wav2vec2 emotion model")
                self._loaded = True
            except Exception as e:
                print(f"[EmotionModel] Could not load full model: {e}")
                self._failed = True
        except Exception as e:
            print(f"[EmotionModel] Load failed: {e}")
            self._failed = True

    @torch.no_grad()
    def predict(self, waveform: np.ndarray, sr: int) -> tuple[float, float, float]:
        """
        Returns (arousal, dominance, valence) in [0, 1].
        waveform: mono float32 array at any sample rate (will be resampled to 16kHz)
        """
        self._load()

        if not self._loaded:
            return self._heuristic(waveform, sr)

        # Resample to 16kHz
        if sr != 16000:
            waveform_t = torch.tensor(waveform).float().unsqueeze(0)
            resamp = T.Resample(sr, 16000)
            waveform = resamp(waveform_t).squeeze(0).numpy()

        inputs = self._processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        out = self._model(input_values)  # (1, 3) → [arousal, dominance, valence]
        a, d, v = out[0].cpu().tolist()
        return float(a), float(d), float(v)

    def _heuristic(self, waveform: np.ndarray, sr: int) -> tuple[float, float, float]:
        """
        Librosa-based heuristic when the model is unavailable.
        Arousal ← RMS energy + spectral centroid (proxy for intensity)
        Valence ← spectral brightness relative to expected range
        Dominance ← zero-crossing rate (low = sustained, high = noisy/percussive)
        """
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        rms_norm = min(rms / 0.1, 1.0)

        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr).mean()
        centroid_norm = float(min(centroid / 4000.0, 1.0))

        zcr = librosa.feature.zero_crossing_rate(waveform).mean()
        zcr_norm = float(min(zcr / 0.2, 1.0))

        arousal   = float(np.clip(rms_norm * 0.6 + centroid_norm * 0.4, 0, 1))
        valence   = float(np.clip(0.3 + centroid_norm * 0.4 - rms_norm * 0.1, 0, 1))
        dominance = float(np.clip(rms_norm * 0.5 + (1.0 - zcr_norm) * 0.5, 0, 1))
        return arousal, dominance, valence


# ─────────────────────────────────────────────────────────────────────────────
#  AudioPipeline
# ─────────────────────────────────────────────────────────────────────────────

class AudioPipeline:
    """
    Reads an audio file and continuously produces AudioFeatures at two rates:
      - Fast features (FFT, onset, energy): every `fast_hop_sec` seconds
      - Emotion update: every `emotion_interval_sec` seconds

    Runs a background thread that reads ahead; get_latest() is non-blocking.
    """

    def __init__(
        self,
        audio_path: str,
        fast_hop_sec:          float = 0.1,    # feature update rate
        emotion_interval_sec:  float = 2.0,    # emotion model update rate
        fft_window_sec:        float = 0.5,    # analysis window for FFT
        emotion_window_sec:    float = 3.0,    # audio window fed to emotion model
        device:                str   = "cpu",
    ):
        self.audio_path           = audio_path
        self.fast_hop_sec         = fast_hop_sec
        self.emotion_interval_sec = emotion_interval_sec
        self.fft_window_sec       = fft_window_sec
        self.emotion_window_sec   = emotion_window_sec

        # Load audio — use soundfile directly (avoids torchaudio codec dependencies)
        print(f"[AudioPipeline] Loading {audio_path} ...")
        try:
            import soundfile as sf
            raw, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            # raw shape: (samples, channels)
            waveform = torch.tensor(raw.T)   # (channels, samples)
        except Exception:
            # Fallback: try librosa (handles mp3 via audioread if ffmpeg available)
            import librosa
            raw, sr = librosa.load(audio_path, sr=None, mono=False)
            if raw.ndim == 1:
                raw = raw[np.newaxis, :]
            waveform = torch.tensor(raw)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        self._waveform = waveform.squeeze(0).numpy().astype(np.float32)
        self._sr = sr
        self._duration = len(self._waveform) / sr
        print(f"[AudioPipeline] Loaded {self._duration:.1f}s at {sr}Hz")

        # Pre-compute mel spectrogram for the whole file (fast at load time)
        self._mel_spec = self._compute_mel_spectrogram()
        self._onset_env = self._compute_onset_envelope()
        self._bpm, self._beat_frames = self._estimate_tempo()

        # State
        self._current_features = AudioFeatures()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pos_sec = 0.0
        self._last_emotion_t = -999.0
        self._cached_emotion = (0.5, 0.5, 0.5)  # (arousal, dom, valence)

        # Emotion model (lazy-loaded)
        self._emotion = EmotionModel(device=device)

    # ── Precomputation ───────────────────────────────────────────────────────

    def _compute_mel_spectrogram(self) -> np.ndarray:
        """Returns (n_mels, T) log-mel spectrogram, normalised to [0,1]."""
        mel = librosa.feature.melspectrogram(
            y=self._waveform,
            sr=self._sr,
            n_mels=FFT_BINS,
            fmax=8000,
            hop_length=int(self._sr * self.fast_hop_sec),
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalise to [0,1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.astype(np.float32)  # (n_mels, T)

    def _compute_onset_envelope(self) -> np.ndarray:
        """Onset strength envelope, one value per fast-hop."""
        hop = int(self._sr * self.fast_hop_sec)
        env = librosa.onset.onset_strength(y=self._waveform, sr=self._sr, hop_length=hop)
        env = env / (env.max() + 1e-8)
        return env.astype(np.float32)

    def _estimate_tempo(self) -> tuple[float, np.ndarray]:
        """Returns (bpm, beat_frame_indices)."""
        hop = int(self._sr * self.fast_hop_sec)
        tempo, beats = librosa.beat.beat_track(
            y=self._waveform, sr=self._sr, hop_length=hop
        )
        # librosa >= 0.10 returns tempo as a 1-element ndarray
        bpm = float(np.atleast_1d(tempo)[0])
        return bpm, beats

    # ── Background thread ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background feature-extraction thread (real-time playback sim)."""
        self._running = True
        self._start_wall = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[AudioPipeline] Started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_latest(self) -> AudioFeatures:
        with self._lock:
            import copy
            return copy.copy(self._current_features)

    def _run(self) -> None:
        """
        Advances playback position in real-time and updates features.
        Simulates what would happen if the audio were playing live.
        """
        hop = int(self._sr * self.fast_hop_sec)
        T_frames = self._mel_spec.shape[1]

        while self._running:
            t_wall = time.time() - self._start_wall
            pos_sec = t_wall % self._duration   # loop the audio
            frame_idx = min(int(pos_sec / self.fast_hop_sec), T_frames - 1)

            # Fast features from precomputed arrays
            mel = self._mel_spec[:, frame_idx].copy()
            onset = float(self._onset_env[min(frame_idx, len(self._onset_env)-1)])

            # RMS from raw waveform window
            s = int(pos_sec * self._sr)
            e = min(s + int(self.fft_window_sec * self._sr), len(self._waveform))
            chunk = self._waveform[s:e] if e > s else np.zeros(1024, dtype=np.float32)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_norm = min(rms / 0.1, 1.0)

            # Spectral features
            spec_c = float(librosa.feature.spectral_centroid(y=chunk, sr=self._sr).mean())
            spec_b = float(librosa.feature.spectral_bandwidth(y=chunk, sr=self._sr).mean())
            spec_r = float(librosa.feature.spectral_rolloff(y=chunk, sr=self._sr, roll_percent=0.85).mean())
            nyq = self._sr / 2.0
            spec_c_n = min(spec_c / nyq, 1.0)
            spec_b_n = min(spec_b / nyq, 1.0)
            spec_r_n = min(spec_r / nyq, 1.0)

            # Beat phase
            beat_phase = self._beat_phase(pos_sec)

            # Emotion update (slow path, every emotion_interval_sec)
            emotion_updated = False
            if (pos_sec - self._last_emotion_t) >= self.emotion_interval_sec or \
               self._last_emotion_t < 0:
                emotion_chunk_len = int(self.emotion_window_sec * self._sr)
                emo_chunk = self._waveform[s: min(s + emotion_chunk_len, len(self._waveform))]
                if len(emo_chunk) > self._sr // 4:   # at least 250ms
                    a, d, v = self._emotion.predict(emo_chunk, self._sr)
                    self._cached_emotion = (a, d, v)
                    self._last_emotion_t = pos_sec
                    emotion_updated = True

            a, d, v = self._cached_emotion

            feat = AudioFeatures(
                mel_spectrum        = mel,
                rms_energy          = rms_norm,
                spectral_centroid   = spec_c_n,
                spectral_bandwidth  = spec_b_n,
                spectral_rolloff    = spec_r_n,
                onset_strength      = onset,
                bpm_estimate        = self._bpm,
                beat_phase          = beat_phase,
                arousal             = a,
                dominance           = d,
                valence             = v,
                pos_sec             = pos_sec,
                emotion_updated     = emotion_updated,
            )

            with self._lock:
                self._current_features = feat

            # Sleep until next hop
            next_t = self._start_wall + (math.floor(t_wall / self.fast_hop_sec) + 1) * self.fast_hop_sec
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    def _beat_phase(self, pos_sec: float) -> float:
        """Estimate fractional position [0,1] within the current beat."""
        beat_dur = 60.0 / max(self._bpm, 1.0)
        return (pos_sec % beat_dur) / beat_dur

    # ── Convenience: full tensor for RL input ────────────────────────────────

    def to_tensor(self, feat: AudioFeatures) -> torch.Tensor:
        """
        Returns a flat float32 tensor:
          [mel_spectrum(128), rms, centroid, bandwidth, rolloff,
           onset, beat_phase, arousal, dominance, valence]
        Total: 138 dimensions
        """
        scalars = np.array([
            feat.rms_energy,
            feat.spectral_centroid,
            feat.spectral_bandwidth,
            feat.spectral_rolloff,
            feat.onset_strength,
            feat.beat_phase,
            feat.arousal,
            feat.dominance,
            feat.valence,
        ], dtype=np.float32)
        return torch.from_numpy(np.concatenate([feat.mel_spectrum, scalars]))


AUDIO_FEAT_DIM = FFT_BINS + 9   # 137


# ─────────────────────────────────────────────────────────────────────────────
#  Offline test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 audio_pipeline.py <audio_file>")
        sys.exit(1)
    pipe = AudioPipeline(sys.argv[1])
    pipe.start()
    for _ in range(10):
        time.sleep(0.5)
        f = pipe.get_latest()
        print(f"pos={f.pos_sec:.1f}s  rms={f.rms_energy:.3f}  "
              f"onset={f.onset_strength:.3f}  "
              f"A={f.arousal:.2f} D={f.dominance:.2f} V={f.valence:.2f}  "
              f"beat={f.beat_phase:.2f}")
    pipe.stop()
