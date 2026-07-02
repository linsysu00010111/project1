"""
End-to-end test: pick-and-place with live MuJoCo viewer + scoring.

Usage::

    python test.py                                    # IK controller (default)
    python test.py --controller ik                    # IK controller
    python test.py --controller nn --checkpoint path  # Trained NN controller
    python test.py --rand                             # randomise block position
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from controllers.base_controller import BaseController
from controllers.ik_controller import IKController
from controllers.nn_controller import (
    ACT_DIM,
    OBS_DIM,
    ActorCritic,
    NNController,
    build_observation,
    scale_action,
)
from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv

# ---------------------------------------------------------------------------
MAX_STEPS: int = 50000
FULL_SCORE_STEPS: int = 2500


# ---------------------------------------------------------------------------
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
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick-and-place test with MuJoCo viewer.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="ik",
        choices=["ik", "nn"],
        help="Controller type (ik or nn).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to NN controller checkpoint (.pt file).",
    )
    parser.add_argument(
        "--rand",
        action="store_true",
        default=False,
        help="Randomise block initial position.",
    )
    args = parser.parse_args()

    # ---- environment ----------------------------------------------------
    env = FrankaEnv(randomize_block=args.rand)

    # ---- controller -----------------------------------------------------
    if args.controller == "ik":
        controller: BaseController = IKController(env)
    else:
        if args.checkpoint is None:
            print("Error: --checkpoint is required when using nn controller.")
            return
        project_root = Path(PROJECT_ROOT_DIR)
        ckpt_path =Path(f"{project_root}/checkpoints/{args.checkpoint}")

        if not ckpt_path.exists():
            print(f"Error: checkpoint not found: {ckpt_path}")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        policy = ActorCritic(OBS_DIM, ACT_DIM, hidden=64).to(device)
        policy.load_state_dict(state["policy"])
        policy.eval()
        controller = NNController(env, policy=policy, device=device)

    # ---- header ---------------------------------------------------------
    print("=" * 55)
    print(f"  Controller:         {args.controller.upper()}")
    print(f"  Block randomisation: {'on' if args.rand else 'off'}")
    print("  Table1 -> Table2  (short -> tall)")
    print("=" * 55)
    print(f"  Initial distance to target: {env.distance_to_target:.3f} m")
    print("  Close the viewer window or press 'q' to finish.\n")

    start_time = time.perf_counter()

    if args.controller == "ik":
        _run_with_viewer(env, controller)
    else:
        _run_with_viewer_nn(env, controller)

    elapsed = time.perf_counter() - start_time
    result = compute_score(env.distance_to_target, controller.step_count)

    print()
    print("-" * 55)
    print("  Results")
    print("-" * 55)
    print(f"  Steps taken:         {result['steps']}")
    print(f"  Wall-clock time:     {elapsed:.2f} s")
    print(f"  Final block-target:  {result['final_distance']:.4f} m")
    print(f"  ---------------------")
    print(f"  Distance score:      {result['distance_score']:.1f} / 50")
    print(f"  Time score:          {result['time_score']:.1f} / 50")
    print(f"  TOTAL SCORE:         {result['total_score']:.1f} / 100")
    print("=" * 55)


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
                print(f"  [{controller.step_count:4d}] -> DONE")
                for _ in range(300):
                    env.step()
                    viewer.sync()
                break

            if controller.step_count >= MAX_STEPS:
                print(f"  !  Timeout after {MAX_STEPS} steps.")
                break


# ---------------------------------------------------------------------------
def _run_with_viewer_nn(env: FrankaEnv, controller: NNController) -> None:
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat = np.array([0.3, 0, 0.2])

        print(f"  Stage: {controller.stage}  {STAGE_NAMES[controller.stage]}")
        while viewer.is_running():
            obs = build_observation(env)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(controller._device)
            with torch.no_grad():
                action_t, _, _ = controller.policy.get_action(obs_t, deterministic=True)
            action_np = action_t.squeeze(0).cpu().numpy()
            scaled = scale_action(action_np)

            env.set_arm_target(scaled[:4])
            env.set_gripper(scaled[4])

            controller._prev_stage = controller.stage
            controller.stage = controller.detect_stage()
            if controller.stage < controller._prev_stage:
                controller.stage = controller._prev_stage

            controller._step_counter += 1
            env.step()
            viewer.sync()

            if controller._step_counter % 100 == 0:
                print(
                    f"  [{controller._step_counter:4d}] "
                    f"Stage: {controller.stage} {STAGE_NAMES[controller.stage]}  "
                    f"Dist: {env.distance_to_target:.3f}m  "
                    f"Grasping: {env.is_grasping}"
                )

            if controller.is_done():
                print(f"  [{controller._step_counter:4d}] -> DONE")
                for _ in range(300):
                    env.step()
                    viewer.sync()
                break

            if controller._step_counter >= MAX_STEPS:
                print(f"  !  Timeout after {MAX_STEPS} steps.")
                break

    print(f"  Final stage: {controller.stage} {STAGE_NAMES[controller.stage]}")
    print(f"  Final distance to target: {env.distance_to_target:.4f} m")


# ---------------------------------------------------------------------------
STAGE_NAMES = ["Approach", "Descend", "Lift", "Transport", "Place", "Done"]


if __name__ == "__main__":
    main()
