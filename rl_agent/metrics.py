"""
metrics.py
──────────
Thin MLflow wrapper for the VJ RL agent.

All logging goes through this module so that:
  - The MLflow run lifecycle is owned in one place
  - Every metric has a consistent naming scheme:  {level}/{category}/{name}
  - The system degrades gracefully if MLflow is unavailable
  - Step counters per level are tracked here so callers don't have to

Naming scheme
─────────────
  low/reward/{component}      low-level per-step reward breakdown
  low/ppo/{stat}              PPO update statistics
  low/audio/{feature}         audio features at step time
  low/visual/{feature}        visual emotion at step time
  low/action/{name}           action histogram (logged as counts)

  high/reward/{component}     high-level per-step reward
  high/ppo/{stat}             high-level PPO stats
  high/goal/{name}            goal selection histogram

  pipeline/{metric}           engine-side metrics (frame time, active passes, …)
  audio/emotion/{dim}         audio emotion stream (arousal/valence/dominance)
  visual/emotion/{dim}        visual emotion stream

Usage:
  from metrics import Metrics
  m = Metrics(experiment="vj_run_1", tracking_uri="mlruns")
  m.start_run(tags={"audio": "djset.mp3"})

  # in low-level step:
  m.log_low_reward({"emotion": 1.2, "beat_sync": 0.5, ...})
  m.log_low_ppo({"policy_loss": 0.01, "value_loss": 0.3, "entropy": 2.1})
  m.log_audio(audio_features)
  m.log_visual(arousal=0.3, valence=-0.1)
  m.log_action("low", action_label)

  # in high-level step:
  m.log_high_reward({"source": 2.0, "flicker": -0.5, ...})
  m.log_goal(goal_name)

  m.end_run()
"""

from __future__ import annotations
import time
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(
        self,
        experiment:    str = "vj_agent",
        tracking_uri:  str = "mlruns",
        disabled:      bool = False,
    ):
        self._disabled = disabled
        self._run = None
        self._experiment = experiment
        self._tracking_uri = tracking_uri

        # Per-level step counters (independent of PPO update count)
        self._step: dict[str, int] = {"low": 0, "high": 0, "pipeline": 0, "audio": 0}

        # Action counters for histograms (flushed every N steps)
        self._action_counts: dict[str, dict[str, int]] = {"low": {}, "high": {}}

        if not disabled:
            try:
                import mlflow
                self._mlflow = mlflow
            except ImportError:
                print("[Metrics] mlflow not installed — logging disabled")
                self._disabled = True

    # ── Run lifecycle ────────────────────────────────────────────────────────

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None) -> None:
        if self._disabled:
            return
        self._mlflow.set_tracking_uri(self._tracking_uri)
        self._mlflow.set_experiment(self._experiment)
        self._run = self._mlflow.start_run(run_name=run_name, tags=tags or {})
        print(f"[Metrics] MLflow run started: {self._run.info.run_id}")

    def end_run(self) -> None:
        if self._disabled or self._run is None:
            return
        # Flush action histograms
        for level in ("low", "high"):
            for label, count in self._action_counts[level].items():
                self._log(f"{level}/action/{label}", count,
                          step=self._step[level])
        self._mlflow.end_run()
        print("[Metrics] MLflow run ended.")

    # ── Core log primitive ────────────────────────────────────────────────────

    def _log(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._disabled or self._run is None:
            return
        try:
            self._mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            # Never crash the agent because of a logging error
            pass

    def _log_dict(self, prefix: str, d: dict, step: int) -> None:
        for k, v in d.items():
            try:
                self._log(f"{prefix}/{k}", float(v), step=step)
            except (TypeError, ValueError):
                pass

    # ── Low-level logging ────────────────────────────────────────────────────

    def log_low_reward(self, components: dict, total: float) -> None:
        s = self._step["low"]
        self._log("low/reward/total", total, s)
        self._log_dict("low/reward", components, s)

    def log_low_ppo(self, info: dict) -> None:
        s = self._step["low"]
        self._log_dict("low/ppo", info, s)
        # Running return estimate
        if "returns_mean" in info:
            self._log("low/ppo/returns_mean", info["returns_mean"], s)

    def log_action(self, level: str, label: str) -> None:
        """Accumulates action counts; logged as a rate per step."""
        counts = self._action_counts.get(level, {})
        counts[label] = counts.get(label, 0) + 1
        self._action_counts[level] = counts
        # Also log one-hot for timeline view
        self._log(f"{level}/action/taken", 1.0, self._step[level])
        self._log(f"{level}/action/index",
                  list(counts.keys()).index(label) if label in counts else 0,
                  self._step[level])

    def tick_low(self) -> None:
        """Increment low-level step counter. Call once per agent step."""
        self._step["low"] += 1

    # ── High-level logging ────────────────────────────────────────────────────

    def log_high_reward(self, components: dict, total: float) -> None:
        s = self._step["high"]
        self._log("high/reward/total", total, s)
        self._log_dict("high/reward", components, s)

    def log_high_ppo(self, info: dict) -> None:
        self._log_dict("high/ppo", info, self._step["high"])

    def log_goal(self, goal_name: str) -> None:
        s = self._step["high"]
        counts = self._action_counts["high"]
        counts[goal_name] = counts.get(goal_name, 0) + 1
        # Log the distribution as individual gauges — useful for seeing
        # whether the agent is collapsing onto a single goal
        total = max(sum(counts.values()), 1)
        for name, cnt in counts.items():
            self._log(f"high/goal_frac/{name}", cnt / total, s)

    def tick_high(self) -> None:
        self._step["high"] += 1

    # ── Audio features ────────────────────────────────────────────────────────

    def log_audio(self, af) -> None:
        """Log AudioFeatures scalar summary (not the full mel spectrum)."""
        s = self._step["audio"]
        self._log("audio/rms_energy",        af.rms_energy,        s)
        self._log("audio/spectral_centroid",  af.spectral_centroid, s)
        self._log("audio/onset_strength",     af.onset_strength,    s)
        self._log("audio/beat_phase",         af.beat_phase,        s)
        self._log("audio/bpm",                af.bpm_estimate,      s)
        self._log("audio/emotion/arousal",    af.arousal,           s)
        self._log("audio/emotion/dominance",  af.dominance,         s)
        self._log("audio/emotion/valence",    af.valence,           s)
        self._step["audio"] += 1

    # ── Visual emotion ────────────────────────────────────────────────────────

    def log_visual(self, arousal: float, valence: float) -> None:
        s = self._step["low"]   # visual is always associated with low-level step
        self._log("visual/emotion/arousal", arousal, s)
        self._log("visual/emotion/valence", valence, s)
        # Distance in (arousal, valence) space from audio — key alignment metric
        # (logged here as a convenience; also computed inside emotion_reward)

    def log_emotion_alignment(self, audio_arousal: float, audio_valence: float,
                               visual_arousal: float, visual_valence: float) -> None:
        import math
        dist = math.sqrt((audio_arousal - visual_arousal)**2 +
                         (audio_valence - visual_valence)**2)
        self._log("alignment/emotion_dist", dist, self._step["low"])
        self._log("alignment/arousal_gap",  audio_arousal - visual_arousal, self._step["low"])
        self._log("alignment/valence_gap",  audio_valence - visual_valence, self._step["low"])

    # ── Pipeline / engine stats ────────────────────────────────────────────────

    def log_pipeline(self, stats) -> None:
        """Log RenderStats from the engine (frame_time_ms, active_passes, etc.)."""
        s = self._step["pipeline"]
        self._log("pipeline/frame_time_ms",  stats.get("frame_time_ms", 0),  s)
        self._log("pipeline/active_passes",  stats.get("active_passes", 0),  s)
        self._log("pipeline/frame_number",   stats.get("frame_number",  0),  s)
        self._step["pipeline"] += 1

    def log_shader_state(self, active_shaders: list[str]) -> None:
        """One gauge per registered shader: 1.0 if active, 0.0 if not."""
        from shader_registry import ALL_NAMES
        s = self._step["pipeline"]
        active_set = set(active_shaders)
        for name in ALL_NAMES:
            self._log(f"pipeline/shader/{name}", 1.0 if name in active_set else 0.0, s)

    # ── Hyperparameter logging ────────────────────────────────────────────────

    def log_params(self, params: dict) -> None:
        if self._disabled or self._run is None:
            return
        try:
            self._mlflow.log_params(params)
        except Exception:
            pass

    # ── Artifact logging ─────────────────────────────────────────────────────

    def log_checkpoint(self, path: str) -> None:
        """Log a model checkpoint file as an MLflow artifact."""
        if self._disabled or self._run is None:
            return
        try:
            import os
            if os.path.exists(path):
                self._mlflow.log_artifact(path, artifact_path="checkpoints")
        except Exception:
            pass
