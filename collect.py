"""
Data collection for behavioural cloning — IK-controller expert trajectories.

Collects (observation, action) pairs from the IK pick-and-place controller
across many episodes with randomised block positions.  The dataset is saved
as compressed ``.npz`` files, one per episode, plus a metadata summary.

Usage::

    python collect.py                                    # 100 episodes, fixed block
    python collect.py --rand                             # randomise block position
    python collect.py --episodes 500 --rand              # 500 episodes
    python collect.py --episodes 200 --out data/my_run   # custom output dir

Output layout::

    data/
    ├── metadata.npz          # episode index, success flags, block/target pos
    ├── episode_0000.npz      # observations, actions
    ├── episode_0001.npz
    └── ...
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from controllers.ik_controller import IKController, MAX_STEPS as IK_MAX_STEPS
from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv


def _collect_one_episode(
    env: FrankaEnv,
    controller: IKController,
) -> Optional[Dict[str, np.ndarray]]:
    """Run one pick-and-place episode and return (obs, act) arrays.

    Returns ``None`` if the episode times out (failure).
    """
    obs_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []

    for _ in range(IK_MAX_STEPS):
        # ---- build observation (before action) ---------------------------
        """
        下面给出示例观测变量 obs
        可以自行按需拓展
        """
        arm_q = env.arm_joint_positions  # (4,)
        arm_dq = env.arm_joint_velocities  # (4,)
        ee_pos = env.endeffector_position  # (3,)
        blk_pos = env.block_position  # (3,)
        tgt_pos = env.target_position  # (3,)
        finger_pos = env.finger_joint_positions  # (2,)
        # 夹爪开度
        finger_open = float(np.mean(finger_pos))
        # 末端执行器到方块的距离
        dist_ee_block = float(np.linalg.norm(ee_pos - blk_pos))
        # < 0.02 认为夹爪闭合
        is_closed = 1.0 if finger_open < 0.02 else 0.0

        obs = np.concatenate(
            [
                arm_q,
                arm_dq,
                ee_pos,
                blk_pos,
                tgt_pos,
                [finger_open, dist_ee_block, is_closed],
            ]
        )
        obs_list.append(obs.astype(np.float32))

        # ---- compute control (sets ctrl internally) ---------------------
        controller.compute_control()

        # ---- build action (the control just computed) -------------------
        arm_target = np.array(
            [env.data.ctrl[aid] for aid in env._arm_actuator_ids],
            dtype=np.float32,
        )
        grip_cmd = float(env.data.ctrl[env._finger_actuator_id]) / 255.0
        act = np.concatenate([arm_target, [grip_cmd]])
        act_list.append(act.astype(np.float32))

        # ---- step simulation --------------------------------------------
        env.step()

        if controller.is_done():
            break
    else:
        # Timed out — episode is a failure
        return None

    return {
        "observations": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect IK-controller trajectories for behavioural cloning.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: PROJECT_ROOT/data/<timestamp>).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--rand",
        action="store_true",
        default=False,
        help="Randomise block initial position (default: off).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- output directory ------------------------------------------------
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(PROJECT_ROOT_DIR) / "data" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ---- environment & controller ---------------------------------------
    env = FrankaEnv(randomize_block=args.rand)

    # ---- collection loop -------------------------------------------------
    success_count = 0
    fail_count = 0
    total_steps = 0
    start_time = time.perf_counter()

    metadata: Dict[str, list] = {
        "episode": [],
        "success": [],
        "steps": [],
        "block_init_pos": [],
        "target_pos": [],
    }

    for ep in range(args.episodes):
        env.reset()
        controller = IKController(env)

        data = _collect_one_episode(env, controller)

        if data is None:
            fail_count += 1
            status = "✗"
        else:
            success_count += 1
            total_steps += len(data["observations"])

            # Save episode
            ep_path = out_dir / f"episode_{ep:04d}.npz"
            np.savez_compressed(ep_path, **data)

            # Track metadata
            metadata["episode"].append(ep)
            metadata["success"].append(1)
            metadata["steps"].append(len(data["observations"]))
            metadata["block_init_pos"].append(data["observations"][0, 11:14])
            metadata["target_pos"].append(data["observations"][0, 14:17])
            status = "✓"

        elapsed = time.perf_counter() - start_time
        rate = (ep + 1) / elapsed if elapsed > 0 else 0.0
        n_steps = len(data["observations"]) if data else 0
        print(
            f"  [{ep+1:4d}/{args.episodes}] {status}  "
            f"steps={n_steps:4d}  "
            f"success={success_count}  fail={fail_count}  "
            f"{rate:.1f} ep/s",
            flush=True,
        )

    # ---- save metadata ---------------------------------------------------
    elapsed = time.perf_counter() - start_time
    meta_path = out_dir / "metadata.npz"
    np.savez_compressed(
        meta_path,
        episode=np.array(metadata["episode"], dtype=np.int32),
        success=np.array(metadata["success"], dtype=np.int32),
        steps=np.array(metadata["steps"], dtype=np.int32),
        block_init_pos=np.array(metadata["block_init_pos"], dtype=np.float32),
        target_pos=np.array(metadata["target_pos"], dtype=np.float32),
    )

    # ---- summary ---------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Collection complete")
    print(f"  ─────────────────")
    print(f"  Episodes requested:  {args.episodes}")
    print(f"  Successful:          {success_count}")
    print(f"  Failed (timeout):    {fail_count}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Avg steps/episode:   {total_steps / max(success_count, 1):.0f}")
    print(f"  Wall-clock time:     {elapsed:.1f} s")
    print(f"  Episodes/second:     {success_count / elapsed:.2f}")
    print(f"  Output directory:    {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
