"""
PPO training for NNController pick-and-place.

Usage::

    python train.py                                          # train from scratch
    python train.py --resume checkpoints/model_050000.pt     # resume from checkpoint
    python train.py --episodes 100                           # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from controllers.nn_controller import (
    ACT_DIM,
    OBS_DIM,
    ActorCritic,
    NNController,
    PPOBuffer,
    build_observation,
    ppo_update,
    scale_action,
)
from env import PROJECT_ROOT_DIR
from env.franka_env import FrankaEnv

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT = {
    "buffer_size": 2048,
    "batch_size": 64,
    "update_epochs": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "max_steps_per_ep": 2500,
    "eval_episodes": 5,
    "save_interval": 50000,
    "total_steps": 500000,
    "log_interval": 2048,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training for pick-and-place")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT["buffer_size"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT["batch_size"])
    parser.add_argument("--update-epochs", type=int, default=DEFAULT["update_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT["lr"])
    parser.add_argument("--gamma", type=float, default=DEFAULT["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT["gae_lambda"])
    parser.add_argument("--clip-epsilon", type=float, default=DEFAULT["clip_epsilon"])
    parser.add_argument("--value-coef", type=float, default=DEFAULT["value_coef"])
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT["entropy_coef"])
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT["max_grad_norm"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT["max_steps_per_ep"])
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT["eval_episodes"])
    parser.add_argument("--save-interval", type=int, default=DEFAULT["save_interval"])
    parser.add_argument("--total-steps", type=int, default=DEFAULT["total_steps"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--rand", action="store_true", default=False, help="Randomise block position")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate(
    env: FrankaEnv,
    controller: NNController,
    num_episodes: int,
    device: torch.device,
) -> dict:
    controller.policy.eval()
    rewards = []
    successes = 0
    steps_list = []

    for ep in range(num_episodes):
        env.reset()
        controller.reset()
        ep_reward = 0.0

        for _ in range(2500):
            obs = build_observation(env)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_t, _, _ = controller.policy.get_action(obs_t, deterministic=True)
            action_np = action_t.squeeze(0).cpu().numpy()
            scaled = scale_action(action_np)
            env.set_arm_target(scaled[:4])
            env.set_gripper(scaled[4])

            controller._update_stage_tracking()
            reward = controller.compute_reward()
            ep_reward += reward
            controller._step_counter += 1

            env.step()
            if controller.is_done():
                break

        rewards.append(ep_reward)
        steps_list.append(controller.step_count)
        if env.distance_to_target < 0.02:
            successes += 1

    controller.policy.train()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / num_episodes,
        "mean_steps": float(np.mean(steps_list)),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Hyper-parameters: {json.dumps(vars(args), indent=2)}")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(PROJECT_ROOT_DIR) / "checkpoints" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Environment
    env = FrankaEnv(render_mode="headless", randomize_block=args.rand)

    # Policy & controller
    policy = ActorCritic(OBS_DIM, ACT_DIM, hidden=64).to(device)
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(state["policy"])
        start_step = state.get("step", 0)
        print(f"  Loaded step={start_step}")

    controller = NNController(env, policy=policy, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = PPOBuffer(args.buffer_size, OBS_DIM, ACT_DIM)

    if args.resume and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    policy.train()
    log_data = []

    # ---- Training loop ---------------------------------------------------
    obs = build_observation(env)
    env.reset()
    controller.reset()
    episode = 0
    global_step = start_step
    ep_start = time.perf_counter()

    print(f"\n{'Step':>8s}  {'Ep':>4s}  {'Reward':>8s}  {'Len':>5s}  {'Success':>8s}  {'Loss':>8s}  {'Elapsed':>8s}")
    print("-" * 65)

    while global_step < args.total_steps:
        # Collect rollout
        for _ in range(args.buffer_size):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                action_t, logprob_t, value_t = controller.policy.get_action(obs_t)

            action_np = action_t.squeeze(0).cpu().numpy()
            logprob = float(logprob_t.item())
            value = float(value_t.item())
            scaled = scale_action(action_np)

            env.set_arm_target(scaled[:4])
            env.set_gripper(scaled[4])

            controller._update_stage_tracking()
            reward = controller.compute_reward()
            controller.episode_reward += reward
            controller._step_counter += 1

            env.step()
            global_step += 1

            next_obs = build_observation(env)
            done = controller.is_done()

            buffer.store(obs, action_np, logprob, reward, done, value)
            obs = next_obs

            if done:
                episode += 1
                ep_rew = controller.episode_reward
                ep_len = controller.step_count
                success = 1.0 if env.distance_to_target < 0.02 else 0.0
                elapsed = time.perf_counter() - ep_start

                if episode % 10 == 0:
                    print(
                        f"{global_step:>8d}  {episode:>4d}  "
                        f"{ep_rew:>8.1f}  {ep_len:>5d}  "
                        f"{success:>8.1f}  {'-':>8s}  "
                        f"{elapsed:>6.1f}s"
                    )

                log_data.append({
                    "step": global_step,
                    "episode": episode,
                    "reward": round(ep_rew, 2),
                    "length": ep_len,
                    "success": success,
                    "time": round(elapsed, 2),
                })

                env.reset()
                controller.reset()
                obs = build_observation(env)
                ep_start = time.perf_counter()

        # PPO update
        update_info = ppo_update(
            policy,
            optimizer,
            buffer,
            device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            update_epochs=args.update_epochs,
            batch_size=args.batch_size,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )
        buffer.clear()
        print(
            f"{global_step:>8d}  {'---':>4s}  {'---':>8s}  {'---':>5s}  "
            f"{'---':>8s}  {update_info['loss']:>8.4f}  {'---':>8s}"
        )

        # Save checkpoint
        if global_step % args.save_interval < args.buffer_size:
            ckpt_path = out_dir / f"model_{global_step:06d}.pt"
            torch.save(
                {
                    "step": global_step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

            # Evaluation
            eval_env = FrankaEnv(render_mode="headless", randomize_block=True)
            eval_controller = NNController(eval_env, policy=policy, device=device)
            eval_results = evaluate(eval_env, eval_controller, args.eval_episodes, device)
            print(
                f"  Eval: reward={eval_results['mean_reward']:.2f}±{eval_results['std_reward']:.2f}  "
                f"success={eval_results['success_rate']:.0%}  "
                f"steps={eval_results['mean_steps']:.0f}"
            )
            log_data[-1] if log_data else {}
            log_data.append({
                "step": global_step,
                "eval_reward": round(eval_results["mean_reward"], 2),
                "eval_success": round(eval_results["success_rate"], 3),
                "eval_steps": round(eval_results["mean_steps"], 1),
            })

    # Save final model
    final_path = out_dir / "model_final.pt"
    torch.save(
        {
            "step": global_step,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    print(f"\nFinal model saved: {final_path}")

    # Save log
    log_path = out_dir / "log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved: {log_path}")

    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
