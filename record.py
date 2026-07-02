"""
Record IK pick-and-place simulation to video.

Usage::

    python record.py                     # headless → video.mp4
    python record.py --video demo.mp4    # custom output path
    python record.py --viewer            # interactive viewer (no recording)
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from env.franka_env import FrankaEnv
from controllers.ik_controller import IKController

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MAX_STEPS: int = 3000
VIDEO_PATH: str = "outputs/video.mp4"
VIDEO_FPS: int = 50  # playback frame rate
VIDEO_SUBSAMPLE: int = 5  # record 1 frame every N sim steps
VIDEO_WIDTH: int = 640
VIDEO_HEIGHT: int = 480


# ---------------------------------------------------------------------------
# Scoring
FULL_SCORE_STEPS: int = 2500  # ≤ 5 s at 0.002 s timestep


def compute_score(final_distance: float, elapsed_steps: int) -> dict:
    dist_score = max(0.0, (1.0 - final_distance / 0.1) * 50.0)
    if elapsed_steps <= FULL_SCORE_STEPS:
        time_score = 50.0
    else:
        time_score = max(
            0.0,
            50.0
            * (
                1.0
                - (elapsed_steps - FULL_SCORE_STEPS) / (MAX_STEPS - FULL_SCORE_STEPS)
            ),
        )
    return {
        "final_distance": round(final_distance, 5),
        "steps": elapsed_steps,
        "distance_score": round(dist_score, 2),
        "time_score": round(time_score, 2),
        "total_score": round(dist_score + time_score, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Record IK pick-and-place to video")
    parser.add_argument(
        "--viewer", action="store_true", help="Launch interactive viewer instead"
    )
    parser.add_argument(
        "--video", type=str, default=VIDEO_PATH, help="Output video path"
    )
    args = parser.parse_args()

    env = FrankaEnv()
    controller = IKController(env)

    print("=" * 55)
    print("  IK Pick-and-Place — Record")
    print("=" * 55)

    start_time = time.perf_counter()

    if args.viewer:
        _run_with_viewer(env, controller)
    else:
        print(f"  Recording to: {args.video}\n")
        _run_recorded(env, controller, video_path=args.video)

    elapsed = time.perf_counter() - start_time
    result = compute_score(env.distance_to_target, controller.step_count)

    print()
    print("─" * 55)
    print("  Results")
    print("─" * 55)
    print(f"  Final stage:         {controller.stage.name}")
    print(f"  Steps taken:         {result['steps']}")
    print(f"  Wall-clock time:     {elapsed:.2f} s")
    print(f"  Final block-target:  {result['final_distance']:.4f} m")
    print(f"  ─────────────────────")
    print(f"  Distance score:      {result['distance_score']:.1f} / 50")
    print(f"  Time score:          {result['time_score']:.1f} / 50")
    print(f"  TOTAL SCORE:         {result['total_score']:.1f} / 100")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Recorded (offscreen render → video)
# ---------------------------------------------------------------------------
def _run_recorded(
    env: FrankaEnv,
    controller: IKController,
    video_path: str = VIDEO_PATH,
) -> None:
    """Run the simulation headless, capturing every frame to a video file."""
    model, data = env.model, env.data

    # Off-screen renderer + viewer-matched camera
    renderer = mujoco.Renderer(model, VIDEO_HEIGHT, VIDEO_WIDTH)
    cam = _make_camera(azimuth=160, elevation=-25, distance=1.8, lookat=(0.3, 0.0, 0.2))

    frames: list[np.ndarray] = []
    hold_remaining: int = 90  # extra VIDEO frames after DONE
    step_counter: int = 0

    for _ in range(MAX_STEPS):
        if not controller.is_done():
            controller.compute_control()

        env.step()
        step_counter += 1

        # Subsample: record 1 frame every VIDEO_SUBSAMPLE sim steps
        if step_counter % VIDEO_SUBSAMPLE == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        if controller.is_done():
            if hold_remaining == 90:
                print(f"  [{controller.step_count:4d}] → DONE")
            # Hold: keep recording at subsampled rate
            if hold_remaining > 0 and step_counter % VIDEO_SUBSAMPLE == 0:
                hold_remaining -= 1
            if hold_remaining <= 0:
                break
    else:
        print(f"  ⚠  Timeout after {MAX_STEPS} steps — task incomplete.")

    renderer.close()

    # Write video via ffmpeg pipe (H.264, universally playable)
    print(f"\n  Encoding {len(frames)} frames → {video_path} …")
    _encode_video_ffmpeg(frames, video_path, VIDEO_FPS, VIDEO_WIDTH, VIDEO_HEIGHT)
    print(f"  Done: {video_path}  ({Path(video_path).stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Camera helper — returns an MjvCamera matching the viewer's viewpoint
# ---------------------------------------------------------------------------
def _make_camera(
    azimuth: float = 160,
    elevation: float = -25,
    distance: float = 1.8,
    lookat: tuple = (0.3, 0.0, 0.2),
) -> mujoco.MjvCamera:
    """Create an ``MjvCamera`` with the same parameters as the viewer."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = lookat
    return cam


# ---------------------------------------------------------------------------
# ffmpeg video encoding
# ---------------------------------------------------------------------------
def _encode_video_ffmpeg(
    frames: list[np.ndarray],
    path: str,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Pipe RGB frames to ffmpeg and encode as H.264 MP4."""
    # Use system ffmpeg for libx264 (browser-compatible H.264).
    # Conda ffmpeg may have broken OpenH264 — prefer /usr/bin/ffmpeg.
    ffmpeg = r"D:\Program Files\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    if not Path(ffmpeg).exists():
        ffmpeg = "ffmpeg"  # fallback

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",  # stdin
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
def _run_with_viewer(env: FrankaEnv, controller: IKController) -> None:
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat = np.array([0.3, 0, 0.2])

        while viewer.is_running():
            controller.compute_control()
            env.step()
            viewer.sync()

            if controller.is_done():
                print(f"  [{controller.step_count:4d}] → DONE")
                for _ in range(300):
                    env.step()
                    viewer.sync()
                break

            if controller.step_count >= MAX_STEPS:
                print(f"  ⚠  Timeout after {MAX_STEPS} steps.")
                break


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
