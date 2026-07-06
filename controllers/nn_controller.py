from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from env.franka_env import FrankaEnv
from .base_controller import BaseController

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------
STAGE_APPROACH = 0
STAGE_DESCEND = 1
STAGE_LIFT = 2
STAGE_TRANSPORT = 3
STAGE_PLACE = 4
STAGE_DONE = 5

STAGE_NAMES = ["Approach", "Descend", "Lift", "Transport", "Place", "Done"]

# Joint limit ranges from MuJoCo model (active joints 1,2,4,6)
JOINT_RANGES: np.ndarray = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-3.0718, -0.0698],
    [-0.0175, 3.7525],
], dtype=np.float32)

# Observation dimension
OBS_DIM = 21
# Action dimension
ACT_DIM = 5

# Distance threshold for considering a stage goal "reached"
GOAL_REACHED_THRESH = 0.10


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Small actor-critic network for PPO continuous control."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
            log_prob = None
        else:
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)


# ---------------------------------------------------------------------------
# PPO experience buffer
# ---------------------------------------------------------------------------
class PPOBuffer:
    """Rollout buffer for PPO."""

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int):
        self.buffer_size = buffer_size
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.logprobs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        idx = self.ptr % self.buffer_size
        self.obs[idx] = obs
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def get(self, gamma: float, gae_lambda: float, device: torch.device):
        size = self.buffer_size if self.full else self.ptr
        obs = torch.from_numpy(self.obs[:size]).to(device)
        actions = torch.from_numpy(self.actions[:size]).to(device)
        logprobs = torch.from_numpy(self.logprobs[:size]).to(device)

        # Compute GAE and returns
        rewards = self.rewards[:size]
        dones = self.dones[:size]
        values = self.values[:size]

        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(size)):
            next_val = values[t + 1] if t + 1 < size else 0.0
            delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values[:size]

        advantages = torch.from_numpy(advantages).to(device)
        returns = torch.from_numpy(returns).to(device)
        return obs, actions, logprobs, returns, advantages

    def clear(self) -> None:
        self.ptr = 0
        self.full = False


# ---------------------------------------------------------------------------
# Helper: scale action from [-1, 1] to joint ranges
# ---------------------------------------------------------------------------
def scale_action(action: np.ndarray) -> np.ndarray:
    """Convert tanh-normalised action to actual joint/gripper targets.

    action[:4] : joint targets (tanh [-1,1] → joint range)
    action[4]  : gripper target (tanh [-1,1] → [0, 1], 0=closed, 1=open)
    """
    scaled = np.empty(5, dtype=np.float32)
    for i in range(4):
        lo, hi = JOINT_RANGES[i]
        scaled[i] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)
    scaled[4] = (action[4] + 1.0) * 0.5
    scaled[4] = np.clip(scaled[4], 0.0, 1.0)
    return scaled


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def build_observation(env: FrankaEnv) -> np.ndarray:
    """Build 21-dim observation vector from environment state."""
    arm_q = env.arm_joint_positions
    arm_dq = env.arm_joint_velocities
    ee_pos = env.endeffector_position
    blk_pos = env.block_position
    tgt_pos = env.target_position
    finger_pos = env.finger_joint_positions
    finger_open = float(np.mean(finger_pos))
    dist_ee_block = float(np.linalg.norm(ee_pos - blk_pos))
    is_grasping = 1.0 if env.is_grasping else 0.0
    dist_block_target = float(np.linalg.norm(blk_pos - tgt_pos))
    return np.concatenate([
        arm_q, arm_dq, ee_pos, blk_pos, tgt_pos,
        [finger_open, dist_ee_block, is_grasping, dist_block_target],
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# NN Controller (PPO policy)
# ---------------------------------------------------------------------------
class NNController(BaseController):
    """Reinforcement-learning controller using PPO.

    Progressive stage detection: stage = number of consecutive sub-goals reached.
    Each stage has a step budget; reward is cut off after exceeding it.
    Path straightness is rewarded for stages 1-4.
    """

    def __init__(
        self,
        env: FrankaEnv,
        policy: Optional[ActorCritic] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.env = env
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self._device) if policy is not None else ActorCritic().to(self._device)

        self.stage: int = STAGE_APPROACH
        self._prev_stage: int = STAGE_APPROACH

        self._prev_goal_dist: float = 0.0
        self.episode_reward: float = 0.0

        # Cache initial scene geometry at each reset
        self._init_blk_pos: np.ndarray = np.zeros(3)
        self._init_tgt_pos: np.ndarray = np.zeros(3)
        self._init_bsz: np.ndarray = np.zeros(3)
        self._init_t2_top: np.ndarray = np.zeros(3)
        self._init_tbl_top: np.ndarray = np.zeros(3)

        # Steps spent in each stage (index 0-4)
        self._stage_step_counter: List[int] = [0] * 5
        # EE position when first entering each stage (for line-deviation check)
        self._stage_entry_pos: List[Optional[np.ndarray]] = [None] * 5

        # Success tracking
        self._success_given: bool = False

    # ------------------------------------------------------------------
    # Sub-goal for each stage (uses cached initial positions)
    # ------------------------------------------------------------------
    def _stage_goal(self, stage: int) -> np.ndarray:
        blk0 = self._init_blk_pos
        tgt = self._init_tgt_pos
        bsz = self._init_bsz
        t2_top = self._init_t2_top
        tbl_top = self._init_tbl_top

        if stage == STAGE_APPROACH:
            return np.array([blk0[0], blk0[1], blk0[2] + bsz[2] * 5.0])
        elif stage == STAGE_DESCEND:
            return np.array([blk0[0], blk0[1], blk0[2] + bsz[2] * 0.5])
        elif stage == STAGE_LIFT:
            lift_amt = (t2_top[2] - tbl_top[2]) + bsz[2] * 5.0
            return np.array([blk0[0], blk0[1], blk0[2] + bsz[2] * 0.5 + lift_amt])
        elif stage == STAGE_TRANSPORT:
            lift_amt = (t2_top[2] - tbl_top[2]) + bsz[2] * 5.0
            return np.array([tgt[0], tgt[1], blk0[2] + bsz[2] * 0.5 + lift_amt])
        elif stage == STAGE_PLACE:
            return np.array([tgt[0], tgt[1], tgt[2] + bsz[2] * 1.3])
        else:
            return tgt.copy()

    # ------------------------------------------------------------------
    # Progressive stage detection
    # ------------------------------------------------------------------
    def detect_stage(self) -> int:
        """Return consecutive sub-goals reached (0-4). DONE is set by success."""
        ee = self.env.endeffector_position
        for i in range(5):
            goal = self._stage_goal(i)
            dist = float(np.linalg.norm(ee - goal))
            if dist > GOAL_REACHED_THRESH:
                return i
        return STAGE_PLACE

    # ------------------------------------------------------------------
    # Perpendicular distance from point to line segment
    # ------------------------------------------------------------------
    @staticmethod
    def _point_line_dist(
        p: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> float:
        ab = b - a
        ap = p - a
        t = float(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-10))
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    # ------------------------------------------------------------------
    # Stage tracking update (shared by compute_control and training loop)
    # ------------------------------------------------------------------
    def _update_stage_tracking(self) -> None:
        self._prev_stage = self.stage
        new_stage = self.detect_stage()
        if new_stage > self.stage:
            for s in range(self.stage, new_stage):
                if self._stage_entry_pos[s] is None:
                    self._stage_entry_pos[s] = self.env.endeffector_position.copy()
            self.stage = new_stage
            st = min(self.stage, 4)
            self._prev_goal_dist = float(np.linalg.norm(
                self.env.endeffector_position - self._stage_goal(st)
            ))
        inc_stage = min(self.stage, 4)
        self._stage_step_counter[inc_stage] += 1

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def compute_reward(self) -> float:
        if self.is_done():
            return 0.0

        stage = self.stage
        prev_stage = self._prev_stage
        ee = self.env.endeffector_position

        # Check for task success (block at target, gripper open)
        if not self.env.is_grasping and stage >= STAGE_PLACE:
            blk = self.env.block_position
            tgt = self.env.target_position
            if float(np.linalg.norm(blk - tgt)) < 0.02:
                self._success_given = True
                self.stage = STAGE_DONE
                return 200.0

        reward = 0.0

        # Stage completion bonus
        if stage > prev_stage:
            reward += 50.0

        cs = min(stage, 4)

        # Progress reward (always on, never cut off)
        goal = self._stage_goal(cs)
        dist = float(np.linalg.norm(ee - goal))
        progress = self._prev_goal_dist - dist
        self._prev_goal_dist = dist

        if progress > 0:
            reward += progress * 30.0

        # Path straightness (gentle)
        if cs >= 1 and self._stage_entry_pos[cs] is not None:
            start = self._stage_entry_pos[cs]
            end = self._stage_goal(cs)
            deviation = self._point_line_dist(ee, start, end)
            reward -= min(deviation * 1.0, 0.5)

        # Grasp bonus
        if self.env.is_grasping and not self._grasp_bonus_given:
            reward += 30.0
            self._grasp_bonus_given = True

        return float(reward)

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------
    def compute_control(self) -> None:
        self._step_counter += 1
        obs = build_observation(self.env)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self._device)

        with torch.no_grad():
            action_t, _, value_t = self.policy.get_action(obs_t, deterministic=False)

        action_np = action_t.squeeze(0).cpu().numpy()
        scaled = scale_action(action_np)

        self.env.set_arm_target(scaled[:4])
        self.env.set_gripper(scaled[4])

        self._update_stage_tracking()
        reward = self.compute_reward()
        self.episode_reward += reward

        self._last_obs = obs
        self._last_action = action_np
        self._last_logprob = 0.0
        self._last_value = float(value_t.item())
        self._last_reward = reward

    def get_last_transition(self):
        return (
            self._last_obs,
            self._last_action,
            self._last_logprob,
            self._last_reward,
            self.is_done(),
            self._last_value,
        )

    def is_done(self) -> bool:
        if self._success_given:
            return True
        if self._step_counter >= 2500:
            return True
        return False

    def reset(self) -> None:
        super().reset()
        self.stage = STAGE_APPROACH
        self._prev_stage = STAGE_APPROACH
        self._prev_goal_dist = float(np.linalg.norm(
            self.env.endeffector_position - self._stage_goal(STAGE_APPROACH)
        ))
        self.episode_reward = 0.0
        self._success_given = False
        self._grasp_bonus_given = False

        # Cache initial geometry for stable goal computation
        self._init_blk_pos = self.env.block_position.copy()
        self._init_tgt_pos = self.env.target_position.copy()
        self._init_bsz = self.env.block_size.copy()
        self._init_t2_top = self.env.table2_top_position.copy()
        self._init_tbl_top = self.env.tabletop_position.copy()

        self._stage_step_counter = [0] * 5
        self._stage_entry_pos = [None] * 5
        self._stage_entry_pos[0] = self.env.endeffector_position.copy()

    def get_stage_stats(self) -> dict:
        return {
            "stage": self.stage,
            "stage_name": STAGE_NAMES[self.stage],
            "episode_reward": round(self.episode_reward, 2),
            "step": self._step_counter,
            "stage_steps": self._stage_step_counter,
        }


# ---------------------------------------------------------------------------
# PPO update routine
# ---------------------------------------------------------------------------
def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: PPOBuffer,
    device: torch.device,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 10,
    batch_size: int = 64,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict:
    obs, actions, old_logprobs, returns, advantages = buffer.get(gamma, gae_lambda, device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total = obs.size(0)
    losses = []

    policy.train()
    for _ in range(update_epochs):
        indices = np.arange(total)
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            end = start + batch_size
            batch = indices[start:end]
            b_obs = obs[batch]
            b_act = actions[batch]
            b_old_logp = old_logprobs[batch]
            b_adv = advantages[batch]
            b_ret = returns[batch]

            log_prob, entropy, value = policy.evaluate(b_obs, b_act)

            ratio = torch.exp(log_prob - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, b_ret)

            entropy_loss = entropy.mean()

            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            losses.append({
                "loss": loss.item(),
                "actor": actor_loss.item(),
                "value": value_loss.item(),
                "entropy": entropy_loss.item(),
                "approx_kl": (0.5 * (ratio - 1.0).pow(2).mean()).item(),
            })

    policy.eval()

    avg = {k: float(np.mean([l[k] for l in losses])) for k in losses[0]}
    return avg
