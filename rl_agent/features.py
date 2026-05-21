"""
features.py  (v2 — registry-driven)
────────────────────────────────────
Converts EngineState → fixed-length float32 numpy observation vectors.
All shader-related dimensions are derived from shader_registry, not hardcoded.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from shared_types import EngineState
from shader_registry import (
    ALL_NAMES, ALL_IDX, N_SHADERS,
    MODIFIER_NAMES, MODIFIER_IDX, N_MODIFIERS,
    FEEDBACK_SOURCE_NAMES,
    BY_NAME,
)


def source_progress(state: EngineState) -> float:
    if not state.has_source or state.source_dur_sec <= 0:
        return 0.0
    return min(state.source_pos_sec / state.source_dur_sec, 1.0)


def encode_passes(state: EngineState) -> np.ndarray:
    """
    Two binary vectors concatenated:
      [modifier_bits (N_MODIFIERS)]  — 1.0 if that modifier is enabled
      [feedback_bits (N_FEEDBACK)]   — 1.0 if any feedback_source is enabled
    Using separate vectors makes the role distinction explicit to the network.
    """
    from shader_registry import N_FEEDBACK_SOURCES, FEEDBACK_SOURCE_NAMES

    mod_vec = np.zeros(N_MODIFIERS,        dtype=np.float32)
    fb_vec  = np.zeros(N_FEEDBACK_SOURCES, dtype=np.float32)

    for p in state.passes:
        shader  = p.get("shader",  "") if isinstance(p, dict) else getattr(p, "shader",  "")
        enabled = p.get("enabled", True) if isinstance(p, dict) else getattr(p, "enabled", True)
        if not enabled:
            continue
        if shader in MODIFIER_IDX:
            mod_vec[MODIFIER_IDX[shader]] = 1.0
        for i, fb_name in enumerate(FEEDBACK_SOURCE_NAMES):
            if shader == fb_name:
                fb_vec[i] = 1.0

    return np.concatenate([mod_vec, fb_vec])


def complexity_score(state: EngineState) -> float:
    """
    Weighted sum of active shaders' complexity_weight values, normalised to [0,1].
    Better proxy for visual intensity than a raw pass count.
    """
    max_weight = sum(s.complexity_weight for s in BY_NAME.values())
    if max_weight <= 0:
        return 0.0
    active_weight = 0.0
    for p in state.passes:
        shader  = p.get("shader",  "") if isinstance(p, dict) else getattr(p, "shader",  "")
        enabled = p.get("enabled", True) if isinstance(p, dict) else getattr(p, "enabled", True)
        if enabled and shader in BY_NAME:
            active_weight += BY_NAME[shader].complexity_weight
    return float(min(active_weight / max_weight, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Dimension constants (computed from registry)
# ─────────────────────────────────────────────────────────────────────────────

from shader_registry import N_FEEDBACK_SOURCES

_PASS_BITS_DIM = N_MODIFIERS + N_FEEDBACK_SOURCES

BASE_DIM = (
    5               # crowd: energy, density, pulse, frequency, sentiment
    + 1             # has_source
    + 1             # source_progress
    + 1             # source_near_end
    + 1             # complexity_score (weighted, not raw count)
    + 1             # active_passes / N_SHADERS (raw count normalised)
    + _PASS_BITS_DIM # per-shader enabled bits (modifier + feedback groups)
    + 1             # frame_time_ms normalised
)


def base_features(state: EngineState) -> np.ndarray:
    prog     = source_progress(state)
    near_end = 1.0 if (prog > 0.85 and state.has_source) else 0.0

    crowd = np.array([
        state.energy,
        state.density,
        state.pulse,
        state.frequency,
        float(np.clip((state.sentiment + 1.0) / 2.0, 0.0, 1.0)),
    ], dtype=np.float32)

    source_feats = np.array([
        float(state.has_source),
        prog,
        near_end,
    ], dtype=np.float32)

    pipeline_feats = np.array([
        complexity_score(state),
        state.active_passes / max(N_SHADERS, 1),
    ], dtype=np.float32)

    pass_bits = encode_passes(state)

    perf = np.array([
        min(state.frame_time_ms / 33.3, 3.0),
    ], dtype=np.float32)

    return np.concatenate([crowd, source_feats, pipeline_feats, pass_bits, perf])


# Validate at import time
_test_dim = base_features(EngineState()).shape[0]
assert _test_dim == BASE_DIM, \
    f"BASE_DIM mismatch: declared {BASE_DIM}, got {_test_dim}"


# ─────────────────────────────────────────────────────────────────────────────
#  High-level history buffer
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_LEN    = 8
HIGH_LEVEL_DIM = BASE_DIM * HISTORY_LEN


class HistoryBuffer:
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
