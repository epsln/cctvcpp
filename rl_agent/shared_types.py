"""
shared_types.py
───────────────
Canonical Python-side definitions for the JSON state/command protocol
that the VJ engine reads and writes.

JSON files used:
  vj_state.json    — written by C++ engine every tick, read by agent
  vj_commands.json — written by agent, read and consumed by C++ engine

Both files use atomic write (write to .tmp, rename) to avoid torn reads.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json, os, time

# ─────────────────────────────────────────────────────────────────────────────
#  State (C++ → Python)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PassState:
    id: str
    shader: str
    enabled: bool

@dataclass
class EngineState:
    # Timing
    timestamp:      float = 0.0   # wall clock (seconds)
    frame_number:   int   = 0
    frame_time_ms:  float = 0.0

    # Source
    has_source:     bool  = False  # is a video/image loaded?
    source_path:    str   = ""
    source_pos_sec: float = 0.0   # current playback position
    source_dur_sec: float = 0.0   # total duration (0 if unknown)
    source_near_end:bool  = False  # within last 10% of duration

    # Pipeline
    active_passes:  int   = 0
    passes:         list  = field(default_factory=list)  # list[PassState dicts]

    last_ll_t:      float = 0.0 # time since last low level change
    last_hl_t:      float = 0.0 # time since last highlevel change

    # Crowd
    energy:         float = 0.0
    density:        float = 0.0
    pulse:          float = 0.0
    frequency:      float = 0.0
    sentiment:      float = 0.0

    @staticmethod
    def from_dict(d: dict) -> "EngineState":
        s = EngineState()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


# ─────────────────────────────────────────────────────────────────────────────
#  Commands (Python → C++)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Command:
    type: str           # matches CommandType enum names in C++
    pass_id:      str   = ""
    shader_name:  str   = ""
    position:     int   = -1
    enabled:      bool  = True
    uniform_name: str   = ""
    uniform_type: str   = "float"   # "int"|"float"|"vec2"|"vec3"|"vec4"
    uniform_value: object = 0.0     # scalar or list
    source_path:  str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CommandBatch:
    commands: list = field(default_factory=list)   # list[Command dicts]
    agent_level: str = "low"    # "low" | "high"  — for logging

    def to_dict(self) -> dict:
        return {"commands": self.commands, "agent_level": self.agent_level}


# ─────────────────────────────────────────────────────────────────────────────
#  Atomic JSON I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically via a .tmp rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json_safe(path: str) -> Optional[dict]:
    """Read JSON; return None on any error (missing file, partial write, etc.)."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
