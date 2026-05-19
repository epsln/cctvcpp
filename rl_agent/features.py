"""
features.py
───────────
Converts EngineState → fixed-length float32 numpy observation vectors
for both the low-level and high-level policies.

Feature design principles:
- All values normalised to [0, 1] or [-1, 1]
- Derived features (near_end, flicker rate) encode the behaviours we want to shape
- Both policies share the same base features; the high-level gets a longer
  history window as additional context
"""

from __future__ import annotations
import numpy as np
from collections import deque
from shared_types import EngineState

KNOWN_SHADERS = ["chromatic", "glitch", "color_grade", "bloom", "kaleidoscope", "feedback"]
N_SHADERS = len(KNOWN_SHADERS)
SHADER_IDX = {s: i for i, s in enumerate(KNOWN_SHADERS)}


def source_progress(state: EngineState) -> float:
    """Playback position as fraction [0,1]. 0 if no source or unknown duration."""
    if not state.has_source or state.source_dur_sec <= 0:
        return 0.0
    return min(state.source_pos_sec / state.source_dur_sec, 1.0)


def encode_passes(state: EngineState) -> np.ndarray:
    """Binary vector: which known shaders are currently enabled."""
    vec = np.zeros(N_SHADERS, dtype=np.float32)
    for p in state.passes:
        shader = p.get("shader", "") if isinstance(p, dict) else p.shader
        enabled = p.get("enabled", True) if isinstance(p, dict) else p.enabled
        if shader in SHADER_IDX and enabled:
            vec[SHADER_IDX[shader]] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
#  Base feature vector (shared by both levels)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIM = (
    5           # crowd: energy, density, pulse, frequency, sentiment
    + 1         # has_source (0/1)
    + 1         # source_progress [0,1]
    + 1         # source_near_end (0/1) — within last 15%
    + 1         # active_passes / N_SHADERS  (normalised)
    + N_SHADERS # per-shader enabled bits
    + 1         # frame_time_ms / 33.3  (normalised to 60fps baseline)
)


def base_features(state: EngineState) -> np.ndarray:
    prog = source_progress(state)
    near_end = 1.0 if (prog > 0.85 and state.has_source) else 0.0

    crowd = np.array([
        state.energy,
        state.density,
        state.pulse,
        state.frequency,
        np.clip((state.sentiment + 1.0) / 2.0, 0.0, 1.0),  # [-1,1] → [0,1]
    ], dtype=np.float32)

    source_feats = np.array([
        float(state.has_source),
        prog,
        near_end,
    ], dtype=np.float32)

    pipeline_feats = np.array([
        state.active_passes / max(N_SHADERS, 1),
    ], dtype=np.float32)

    pass_bits = encode_passes(state)

    perf = np.array([
        min(state.frame_time_ms / 33.3, 3.0),  # cap at 3× to avoid huge outliers
    ], dtype=np.float32)

    return np.concatenate([crowd, source_feats, pipeline_feats, pass_bits, perf])


assert BASE_DIM == base_features(EngineState()).shape[0], \
    f"BASE_DIM mismatch: declared {BASE_DIM}, got {base_features(EngineState()).shape[0]}"


# ─────────────────────────────────────────────────────────────────────────────
#  History-augmented features (for high-level policy)
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_LEN    = 8    # how many past states to include
HIGH_LEVEL_DIM = BASE_DIM * HISTORY_LEN


class HistoryBuffer:
    """Rolling window of base feature vectors for the high-level policy."""
    def __init__(self, maxlen: int = HISTORY_LEN):
        self.buf: deque = deque(maxlen=maxlen)
        self.maxlen = maxlen
        null = base_features(EngineState())
        for _ in range(maxlen):
            self.buf.append(null.copy())

    def push(self, state: EngineState) -> None:
        self.buf.append(base_features(state))

    def as_vector(self) -> np.ndarray:
        return np.concatenate(list(self.buf)).astype(np.float32)
