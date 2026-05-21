"""
shader_registry.py
──────────────────
Single source of truth for shader metadata.  Both the C++ engine (via a JSON
sidecar) and all Python components import from here.

Shader roles
────────────
  SOURCE          A shader that generates or replaces the image entirely
                  (e.g. a procedural pattern generator, a video passthrough).
                  The pipeline must have exactly one active SOURCE at position 0.
                  In practice the video decode path IS the source; SOURCE shaders
                  are overlays / replacements that can be swapped in.

  MODIFIER        A shader that transforms the image from the previous stage.
                  Most effects live here (chromatic, glitch, bloom, …).
                  Multiple MODIFIERs can be stacked in any order.

  FEEDBACK_SOURCE A shader that reads its OWN previous output and feeds it back
                  (e.g. the echo/trail feedback shader).  Semantically it acts as
                  a persistent memory layer.  Constraint: must be placed AFTER at
                  least one MODIFIER or SOURCE so it has something to compound on.
                  Only one FEEDBACK_SOURCE should be active at a time (stacking
                  two causes exponential blowup).

  COMPOSITE       A shader that blends two texture streams (future use).

Action slots by role
────────────────────
The agent's action space is derived from the registry, not hardcoded:
  - One TOGGLE action per registered MODIFIER  (slot index = registry order)
  - One TOGGLE action for the FEEDBACK_SOURCE group  (only one allowed active)
  - Video navigation and intensity actions are appended at the end

This means adding a new shader to REGISTRY automatically expands the action
space and observation vector at the next agent restart.  Saved weights are
invalidated when the registry changes (handled by a version hash check).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib, json, os

# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UniformSpec:
    name:         str
    type:         str        # "float" | "int" | "vec2" | "vec3" | "vec4"
    default:      object     # scalar or list
    min_val:      float = 0.0
    max_val:      float = 1.0
    description:  str  = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "default": self.default,
                "min": self.min_val, "max": self.max_val, "desc": self.description}


@dataclass
class ShaderSpec:
    name:          str
    role:          str               # "source" | "modifier" | "feedback_source" | "composite"
    description:   str
    uniforms:      list[UniformSpec] = field(default_factory=list)
    # Ordering hints (soft constraints for the pipeline builder)
    preferred_position: str = "any"  # "first" | "last" | "any"
    # Emotion affinity: which audio emotions this shader is most appropriate for.
    # Used by the high-level policy to bias shader selection.
    # Keys: "high_arousal", "low_arousal", "high_valence", "low_valence"
    emotion_affinity: list[str] = field(default_factory=list)
    # Intensity: does cranking this shader up increase visual complexity?
    complexity_weight: float = 1.0   # multiplied into the visual-intensity proxy

    def uniform_defaults(self) -> dict[str, object]:
        return {u.name: u.default for u in self.uniforms}

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "role":               self.role,
            "description":        self.description,
            "preferred_position": self.preferred_position,
            "emotion_affinity":   self.emotion_affinity,
            "complexity_weight":  self.complexity_weight,
            "uniforms":           [u.to_dict() for u in self.uniforms],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  The registry
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: list[ShaderSpec] = [

    ShaderSpec(
        name        = "chromatic",
        role        = "modifier",
        description = "RGB channel split + barrel distortion. Adds tension and edge.",
        preferred_position = "any",
        emotion_affinity   = ["high_arousal", "low_valence"],
        complexity_weight  = 0.6,
        uniforms    = [
            UniformSpec("uStrength", "float", 0.008, 0.0, 0.05,  "Aberration amount"),
            UniformSpec("uBarrel",   "float", 0.10,  0.0, 0.5,   "Barrel distortion"),
        ],
    ),

    ShaderSpec(
        name        = "edge_detect",
        role        = "modifier",
        description = "Apply a edge detection algorithm",
        preferred_position = "last",   
        emotion_affinity   = ["low_arousal", "high_valence", "low_valence"],
        complexity_weight  = 0.3,
        uniforms    = [
            UniformSpec("uSaturation", "float", 1.3, 0.0, 3.0, "Saturation multiplier"),
            UniformSpec("uContrast",   "float", 1.1, 0.5, 2.5, "Contrast"),
            UniformSpec("uHueShift",   "float", 5.0, -180.0, 180.0, "Hue shift deg/sec"),
        ],
    ),

    ShaderSpec(
        name        = "kaleidoscope",
        role        = "modifier",
        description = "Mirror-segment kaleidoscope.",
        preferred_position = "first",  # works best on the raw source
        emotion_affinity   = ["high_valence", "low_arousal"],
        complexity_weight  = 1.0,
        uniforms    = [
            UniformSpec("uSegments", "float", 6.0, 2.0, 16.0, "Mirror count"),
            UniformSpec("uRotation", "float", 0.2, 0.0, 2.0,  "Rotation speed"),
        ],
    ),

    ShaderSpec(
        name        = "feedback",
        role        = "feedback_source",
        description = "Echo trail: componds the previous frame back with zoom+spin. "
                      "Acts as a persistent memory layer — must not be the only pass.",
        preferred_position = "last",   # compounds whatever came before
        emotion_affinity   = ["low_arousal"],
        complexity_weight  = 1.5,      # can rapidly increase visual entropy
        uniforms    = [
            UniformSpec("uDecay", "float", 0.85, 0.0, 0.99, "Trail persistence"),
            UniformSpec("uZoom",  "float", 0.995,0.9, 1.0,  "Feedback zoom"),
            UniformSpec("uSpin",  "float", 0.3,  0.0, 5.0,  "Rotation deg/frame"),
        ],
    ),

    # ── Placeholder: add new shaders here ────────────────────────────────────
    # ShaderSpec(
    #     name        = "my_new_shader",
    #     role        = "modifier",   # or "source" / "feedback_source" / "composite"
    #     description = "...",
    #     uniforms    = [ UniformSpec("uMyParam", "float", 0.5) ],
    # ),
]

# ─────────────────────────────────────────────────────────────────────────────
#  Derived views (computed once at import time)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered lists by role
MODIFIERS:        list[ShaderSpec] = [s for s in REGISTRY if s.role == "modifier"]
FEEDBACK_SOURCES: list[ShaderSpec] = [s for s in REGISTRY if s.role == "feedback_source"]
SOURCES:          list[ShaderSpec] = [s for s in REGISTRY if s.role == "source"]

# Name → spec lookup
BY_NAME: dict[str, ShaderSpec] = {s.name: s for s in REGISTRY}

# Flat name lists (stable order for obs vector indexing)
MODIFIER_NAMES:        list[str] = [s.name for s in MODIFIERS]
FEEDBACK_SOURCE_NAMES: list[str] = [s.name for s in FEEDBACK_SOURCES]
ALL_NAMES:             list[str] = [s.name for s in REGISTRY]

# Index maps
MODIFIER_IDX:  dict[str, int] = {n: i for i, n in enumerate(MODIFIER_NAMES)}
ALL_IDX:       dict[str, int] = {n: i for i, n in enumerate(ALL_NAMES)}

N_MODIFIERS        = len(MODIFIERS)
N_FEEDBACK_SOURCES = len(FEEDBACK_SOURCES)
N_SHADERS          = len(REGISTRY)


def version_hash() -> str:
    """
    Hash of the registry contents.  If this changes, saved model weights that
    depend on N_SHADERS are invalidated and should be retrained.
    """
    blob = json.dumps([s.to_dict() for s in REGISTRY], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
#  Action space builder (called by agents at init)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionSpec:
    index:       int
    kind:        str   # "noop" | "toggle_modifier" | "toggle_feedback" | "next_video"
                       # | "prev_video" | "intensity_up" | "intensity_down"
    shader_name: Optional[str] = None
    label:       str  = ""


def build_action_space() -> list[ActionSpec]:
    """
    Builds the full action list dynamically from REGISTRY.
    Layout:
      0              NOOP
      1…N_MOD        TOGGLE each MODIFIER  (one per modifier in registry order)
      N_MOD+1        TOGGLE FEEDBACK_SOURCE group  (cycles through feedback shaders)
      N_MOD+2        NEXT_VIDEO
      N_MOD+3        PREV_VIDEO
      N_MOD+4        INTENSITY_UP
      N_MOD+5        INTENSITY_DOWN
    """
    actions = [ActionSpec(0, "noop", label="NOOP")]
    for i, spec in enumerate(MODIFIERS):
        actions.append(ActionSpec(i+1, "toggle_modifier",
                                  shader_name=spec.name,
                                  label=f"TOGGLE_{spec.name.upper()}"))
    base = N_MODIFIERS + 1
    # One slot covers the whole feedback-source group
    fb_names = "/".join(FEEDBACK_SOURCE_NAMES) or "feedback"
    actions.append(ActionSpec(base,   "toggle_feedback", label=f"TOGGLE_{fb_names.upper()}"))
    actions.append(ActionSpec(base+1, "next_video",      label="NEXT_VIDEO"))
    actions.append(ActionSpec(base+2, "prev_video",      label="PREV_VIDEO"))
    actions.append(ActionSpec(base+3, "intensity_up",    label="INTENSITY_UP"))
    actions.append(ActionSpec(base+4, "intensity_down",  label="INTENSITY_DOWN"))
    return actions


ACTION_SPACE: list[ActionSpec] = build_action_space()
N_ACTIONS:    int               = len(ACTION_SPACE)

# Convenience index constants (computed, not hardcoded)
NOOP_IDX          = 0
TOGGLE_MODIFIER_START = 1
TOGGLE_MODIFIER_END   = N_MODIFIERS          # inclusive
TOGGLE_FEEDBACK_IDX   = N_MODIFIERS + 1
NEXT_VIDEO_IDX        = N_MODIFIERS + 2
PREV_VIDEO_IDX        = N_MODIFIERS + 3
INTENSITY_UP_IDX      = N_MODIFIERS + 4
INTENSITY_DOWN_IDX    = N_MODIFIERS + 5

_STRUCTURAL_INDICES = set(range(1, N_MODIFIERS + 4))  # toggles + video switches

def is_structural(action_idx: int) -> bool:
    return action_idx in _STRUCTURAL_INDICES


# ─────────────────────────────────────────────────────────────────────────────
#  Constraint checker: is the current pipeline valid?
# ─────────────────────────────────────────────────────────────────────────────

def validate_pipeline(active_shader_names: list[str]) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Checks:
      - No more than one feedback_source active
      - If feedback_source is active, at least one modifier or source is also active
    """
    active_roles = [BY_NAME[n].role for n in active_shader_names if n in BY_NAME]
    fb_count = active_roles.count("feedback_source")

    if fb_count > 1:
        return False, f"Multiple feedback_source shaders active ({fb_count})"
    if fb_count == 1 and len(active_shader_names) < 2:
        return False, "feedback_source active with no other pass to compound on"
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
#  JSON sidecar (written at startup, read by agent_runner for discovery)
# ─────────────────────────────────────────────────────────────────────────────

def write_registry_json(path: str = "vj_shader_registry.json") -> None:
    data = {
        "version":          version_hash(),
        "n_shaders":        N_SHADERS,
        "n_modifiers":      N_MODIFIERS,
        "n_feedback":       N_FEEDBACK_SOURCES,
        "n_actions":        N_ACTIONS,
        "shaders":          [s.to_dict() for s in REGISTRY],
        "action_space":     [{"index": a.index, "kind": a.kind,
                               "shader": a.shader_name, "label": a.label}
                             for a in ACTION_SPACE],
    }
    import json as _json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"[Registry] Written {path}  (hash={version_hash()}, "
          f"{N_SHADERS} shaders, {N_ACTIONS} actions)")


if __name__ == "__main__":
    write_registry_json()
    print(f"\nAction space ({N_ACTIONS} actions):")
    for a in ACTION_SPACE:
        spec = BY_NAME.get(a.shader_name) if a.shader_name else None
        role = f"  [{spec.role}]" if spec else ""
        print(f"  {a.index:2d}  {a.label:<30s}{role}")
    print(f"\nRegistry version hash: {version_hash()}")
