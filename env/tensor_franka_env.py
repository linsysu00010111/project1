"""
Tensor wrapper for FrankaEnv — converts all state properties to PyTorch tensors.

Usage::

    from env.franka_env import FrankaEnv
    from env.tensor_franka_env import TensorFrankaEnv

    env = FrankaEnv()
    tenv = TensorFrankaEnv(env)

    q = tenv.arm_joint_positions    # torch.Tensor, shape (4,)
    d = tenv.distance_to_target     # torch.Tensor, scalar
    state = tenv.get_state_dict()   # Dict[str, torch.Tensor]

    tenv.step()                     # delegated to env
    tenv.reset()
"""

from typing import Dict

import numpy as np
import torch

from .franka_env import FrankaEnv


class TensorFrankaEnv:
    """Wraps a ``FrankaEnv`` so that state properties return PyTorch tensors.

    All properties and methods are explicitly defined so that IDE code
    completion and type checking work correctly.

    Usage::

        env = FrankaEnv()
        tenv = TensorFrankaEnv(env)
        q = tenv.arm_joint_positions   # torch.Tensor, shape (4,)
        tenv.step()                    # delegated to env.step()
    """

    def __init__(self, env: FrankaEnv) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # Simulation control (explicit delegation)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the underlying simulation."""
        self.env.reset()

    def step(self) -> None:
        """Advance the underlying simulation by one time step."""
        self.env.step()

    @property
    def model(self):
        """MuJoCo model of the underlying environment."""
        return self.env.model

    @property
    def data(self):
        """MuJoCo data of the underlying environment."""
        return self.env.data

    # ------------------------------------------------------------------
    # State properties — scene geometry
    # ------------------------------------------------------------------
    @property
    def table_position(self) -> torch.Tensor:
        """World-frame position of the table body centre."""
        return self._to_tensor(self.env.table_position)

    @property
    def tabletop_position(self) -> torch.Tensor:
        """World-frame position of the top surface centre of the table."""
        return self._to_tensor(self.env.tabletop_position)

    @property
    def table2_position(self) -> torch.Tensor:
        """World-frame position of the taller table (table2) body centre."""
        return self._to_tensor(self.env.table2_position)

    @property
    def table2_top_position(self) -> torch.Tensor:
        """World-frame position of the top surface centre of table2."""
        return self._to_tensor(self.env.table2_top_position)

    @property
    def block_position(self) -> torch.Tensor:
        """World-frame position of the block centre."""
        return self._to_tensor(self.env.block_position)

    @property
    def block_size(self) -> torch.Tensor:
        """Half-extents of the block box geom."""
        return self._to_tensor(self.env.block_size)

    @property
    def target_position(self) -> torch.Tensor:
        """World-frame position of the mocap target (place goal on table2)."""
        return self._to_tensor(self.env.target_position)

    @property
    def distance_to_target(self) -> torch.Tensor:
        """Euclidean distance from block centre to the mocap target."""
        return self._to_tensor(self.env.distance_to_target)

    # ------------------------------------------------------------------
    # State properties — arm
    # ------------------------------------------------------------------
    @property
    def arm_joint_positions(self) -> torch.Tensor:
        """Arm joint angles [rad] for active joints (joint1, joint2, joint4, joint6)."""
        return self._to_tensor(self.env.arm_joint_positions)

    @property
    def arm_joint_velocities(self) -> torch.Tensor:
        """Arm joint velocities [rad/s] for active joints (joint1, joint2, joint4, joint6)."""
        return self._to_tensor(self.env.arm_joint_velocities)

    @property
    def arm_joint_torques(self) -> torch.Tensor:
        """Actuator forces [Nm] for active actuators (actuator1, actuator2, actuator4, actuator6)."""
        return self._to_tensor(self.env.arm_joint_torques)

    # ------------------------------------------------------------------
    # State properties — fingers
    # ------------------------------------------------------------------
    @property
    def finger_joint_positions(self) -> torch.Tensor:
        """Finger joint positions [m] for finger_joint1 & finger_joint2."""
        return self._to_tensor(self.env.finger_joint_positions)

    # ------------------------------------------------------------------
    # State properties — end-effector
    # ------------------------------------------------------------------
    @property
    def endeffector_position(self) -> torch.Tensor:
        """World-frame position of the hand body centre."""
        return self._to_tensor(self.env.endeffector_position)

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return all state properties as a dictionary of tensors."""
        return {
            "table_position": self.table_position,
            "tabletop_position": self.tabletop_position,
            "table2_position": self.table2_position,
            "table2_top_position": self.table2_top_position,
            "block_position": self.block_position,
            "block_size": self.block_size,
            "target_position": self.target_position,
            "distance_to_target": self.distance_to_target,
            "arm_joint_positions": self.arm_joint_positions,
            "finger_joint_positions": self.finger_joint_positions,
            "arm_joint_velocities": self.arm_joint_velocities,
            "arm_joint_torques": self.arm_joint_torques,
            "endeffector_position": self.endeffector_position,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert a NumPy array or scalar to a PyTorch tensor."""
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype)
        return torch.tensor(value, dtype=dtype)
