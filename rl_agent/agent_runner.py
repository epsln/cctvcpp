"""
agent_runner.py  (v3 — shared Metrics, registry sidecar)
"""
import argparse, os, time
from shared_types     import EngineState, CommandBatch, atomic_write_json, read_json_safe
from audio_pipeline   import AudioPipeline, AudioFeatures
from visual_emotion   import VisualEmotionEstimator
from low_level_agent  import LowLevelAgent
from high_level_agent import HighLevelAgent
from metrics          import Metrics
from shader_registry  import write_registry_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio",          required=True)
    p.add_argument("--media",          default="assets")
    p.add_argument("--state",          default="vj_state.json")
    p.add_argument("--commands",       default="vj_commands.json")
    p.add_argument("--checkpoints",    default="checkpoints")
    p.add_argument("--poll_hz",        type=float, default=20.0)
    p.add_argument("--win_x",          type=int,   default=0)
    p.add_argument("--win_y",          type=int,   default=0)
    p.add_argument("--win_w",          type=int,   default=1280)
    p.add_argument("--win_h",          type=int,   default=720)
    p.add_argument("--frame_file",     default="vj_frame.ppm")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--mlflow_uri",     default="mlruns")
    p.add_argument("--experiment",     default="vj_agent")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--no_mlflow",      action="store_true")
    return p.parse_args()


def read_state(path):
    d = read_json_safe(path)
    return EngineState.from_dict(d) if d else None

def write_commands(path, batch):
    atomic_write_json(path, batch.to_dict())


def main():
    args = parse_args()
    os.makedirs(args.checkpoints, exist_ok=True)

    # Write registry sidecar so the C++ side and any external tools can
    # discover shader metadata without importing Python
    write_registry_json("vj_shader_registry.json")

    # ── Shared metrics instance ───────────────────────────────────────────────
    m = Metrics(
        experiment   = args.experiment,
        tracking_uri = args.mlflow_uri,
        disabled     = args.no_mlflow,
    )
    m.start_run(
        run_name = args.run_name or f"vj_{int(time.time())}",
        tags     = {
            "audio":     os.path.basename(args.audio),
            "media_dir": args.media,
        },
    )

    # ── Sub-systems ───────────────────────────────────────────────────────────
    print("[Runner] Initialising audio pipeline ...")
    audio = AudioPipeline(args.audio, device=args.device)

    print("[Runner] Initialising visual emotion estimator ...")
    visual_emo = VisualEmotionEstimator(device=args.device, update_interval_sec=1.5)

    high = HighLevelAgent(
        weights_path  = os.path.join(args.checkpoints, "high_level"),
        device        = args.device,
        metrics       = m,
    )
    low = LowLevelAgent(
        media_dir     = args.media,
        weights_path  = os.path.join(args.checkpoints, "low_level"),
        device        = args.device,
        metrics       = m,
    )

    audio.start()
    print("[Runner] Loop started. Ctrl-C to stop.")

    poll_dt    = 1.0 / args.poll_hz
    last_frame = -1
    state      = EngineState()
    af         = AudioFeatures()
    v_arousal  = 0.0
    v_valence  = 0.0
    pil_frame  = None

    try:
        while True:
            t0 = time.time()

            # 1. Engine state
            s = read_state(args.state)
            if s is not None and s.frame_number != last_frame:
                state      = s
                last_frame = s.frame_number
                high.observe(state, af)
                # Log pipeline stats from engine state
                m.log_pipeline({
                    "frame_time_ms": state.frame_time_ms,
                    "active_passes": state.active_passes,
                    "frame_number":  state.frame_number,
                })

            # 2. Audio
            af = audio.get_latest()

            # 3. Frame capture + visual emotion
            pil_frame = visual_emo.capture_window(
                args.win_x, args.win_y, args.win_w, args.win_h)
            if pil_frame is None:
                pil_frame = visual_emo.capture_from_file(args.frame_file)
            if pil_frame is not None:
                v_arousal, v_valence = visual_emo.estimate(pil_frame)

            # 4. High-level step
            if high.should_step():
                goal = high.step(state, af)
                low.set_goal(goal)
                print(f"[Runner] HL goal → {goal}")

            # 5. Low-level step
            if low.should_step():
                batch = low.step(state, af, v_arousal, v_valence, pil_frame)
                if batch.commands:
                    write_commands(args.commands, batch)
                    print(f"[Runner] LL → "
                          + ", ".join(c.get("type","?") for c in batch.commands))

            elapsed = time.time() - t0
            sleep   = poll_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n[Runner] Saving and ending run ...")
        audio.stop()
        high.trainer.save(high.weights_path_versioned)
        low.trainer.save(low.weights_path_versioned)
        m.end_run()
        print(f"[Runner] Done.  HL={high.total_reward:.1f}  LL={low.total_reward:.1f}")


if __name__ == "__main__":
    main()
