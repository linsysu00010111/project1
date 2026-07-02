"""
MuJoCo environment: Franka Emika Panda arm + two tables + block.

Joints 3, 5, 7 are locked — the arm has 4 active DOF.

Usage::

    from env.franka_env import FrankaEnv

    env = FrankaEnv()
    env.step()
    q = env.arm_joint_positions    # np.ndarray, shape (4,)
    p = env.endeffector_position   # np.ndarray, shape (3,)
    d = env.distance_to_target     # float
    env.reset()
"""

import os
from typing import List

import mujoco
import numpy as np

from env import PROJECT_ROOT_DIR


class FrankaEnv:
    """MuJoCo environment: Franka Emika Panda arm + table + movable block.

    Provides property-based access to simulation state. All position/velocity
    properties return **copies** of the underlying MuJoCo arrays, so callers
    may modify them safely without affecting the simulation.

    Joints 3, 5, 7 are locked at fixed values, reducing the arm to 4 active
    DOF (joints 1, 2, 4, 6).

    Usage::

        env = FrankaEnv()
        env.step()
        q = env.arm_joint_positions  # np.ndarray of shape (4,)
    """

    # ------------------------------------------------------------------
    # Class-level name constants (shared across instances)
    # ------------------------------------------------------------------
    # Active joints (4-DOF) — publicly exposed and controlled.
    # Joints 3, 5, 7 are locked at fixed values to reduce the arm DOF.
    ARM_JOINT_NAMES: tuple = (
        "joint1",
        "joint2",
        "joint4",
        "joint6",
    )
    FINGER_JOINT_NAMES: tuple = ("finger_joint1", "finger_joint2")
    ARM_ACTUATOR_NAMES: tuple = (
        "actuator1",
        "actuator2",
        "actuator4",
        "actuator6",
    )
    FINGER_ACTUATOR_NAME: str = "actuator8"

    # Fixed joints — locked at these values, never exposed to callers.
    FIXED_JOINT_NAMES: tuple = ("joint3", "joint5", "joint7")
    FIXED_ACTUATOR_NAMES: tuple = ("actuator3", "actuator5", "actuator7")
    # Corresponding default values from the original 7-DOF pose:
    #   [joint3=0, joint5=0, joint7=0.785]
    FIXED_JOINT_VALUES: np.ndarray = np.array([0.0, 0.0, 0.785])

    # Initial joint positions: 4 active arm [rad] + 2 finger [m]
    #   joint1=0, joint2=-45°, joint4=-135°, joint6=90°
    DEFAULT_ARM_QPOS: np.ndarray = np.array([0, -0.785, -2.356, 1.571])
    DEFAULT_FINGER_QPOS: np.ndarray = np.array([0.0, 0.0])

    # Block initial pose — centre of table1 top surface
    DEFAULT_BLOCK_POS: np.ndarray = np.array([0.45, 0.16, 0.22])
    DEFAULT_BLOCK_QUAT: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
    # Randomisation range in x, y [m] — block stays on table1
    BLOCK_RANDOM_RANGE: float = 0.06

    # Target mocap position (centre of table2 top surface + half geom height)
    TARGET_POS: np.ndarray = np.array([0.45, -0.16, 0.32])

    # Scene XML (relative to project root)
    SCENE_XML: str = "franka_emika_panda/franka_pick.xml"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self, render_mode: str = "human", randomize_block: bool = False
    ) -> None:
        self.render_mode = render_mode
        self._randomize_block = randomize_block

        xml_path = os.path.join(PROJECT_ROOT_DIR, self.SCENE_XML)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Scene file not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Reliable grasping state
        self._attach_block: bool = False
        self._rel_pos: np.ndarray = np.zeros(3)
        self._rel_rot: np.ndarray = np.eye(3)

        self._cache_ids()
        self.reset()

    # ------------------------------------------------------------------
    # ID caching
    # ------------------------------------------------------------------
    def _cache_ids(self) -> None:
        """Pre-compute every MuJoCo ID used by properties.

        MuJoCo's ``mj_name2id`` does a linear string scan — calling it in
        every property getter is wasteful.  We call it once at init and
        store the integer IDs (and derived qpos/dof addresses).
        """
        model = self.model
        _id = mujoco.mj_name2id  # local alias for brevity

        # ---- bodies ---------------------------------------------------
        self._table_body_id: int = _id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self._table2_body_id: int = _id(model, mujoco.mjtObj.mjOBJ_BODY, "table2")
        self._block_body_id: int = _id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self._hand_body_id: int = _id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # ---- geoms ----------------------------------------------------
        self._table_geom_id: int = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        self._table2_geom_id: int = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table2_geom")
        self._block_geom_id: int = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")

        # ---- arm joints & their qpos / dof addresses (active only) -----
        self._arm_joint_ids: List[int] = [
            _id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ARM_JOINT_NAMES
        ]
        self._arm_qpos_adrs: List[int] = [
            model.jnt_qposadr[j] for j in self._arm_joint_ids
        ]
        self._arm_dof_adrs: List[int] = [
            model.jnt_dofadr[j] for j in self._arm_joint_ids
        ]

        # ---- fixed joints (locked, internal) ----------------------------
        self._fixed_joint_ids: List[int] = [
            _id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.FIXED_JOINT_NAMES
        ]
        self._fixed_qpos_adrs: List[int] = [
            model.jnt_qposadr[j] for j in self._fixed_joint_ids
        ]
        self._fixed_dof_adrs: List[int] = [
            model.jnt_dofadr[j] for j in self._fixed_joint_ids
        ]

        # ---- finger bodies (for contact detection) --------------------
        self._left_finger_body_id: int = _id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self._right_finger_body_id: int = _id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        self._finger_body_ids: set[int] = {
            self._left_finger_body_id,
            self._right_finger_body_id,
        }

        # ---- finger joints --------------------------------------------
        self._finger_joint_ids: List[int] = [
            _id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.FINGER_JOINT_NAMES
        ]
        self._finger_qpos_adrs: List[int] = [
            model.jnt_qposadr[j] for j in self._finger_joint_ids
        ]

        # ---- arm actuators (active) ------------------------------------
        self._arm_actuator_ids: List[int] = [
            _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.ARM_ACTUATOR_NAMES
        ]

        # ---- fixed actuators -------------------------------------------
        self._fixed_actuator_ids: List[int] = [
            _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.FIXED_ACTUATOR_NAMES
        ]

        # ---- finger actuator ------------------------------------------
        self._finger_actuator_id: int = _id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.FINGER_ACTUATOR_NAME
        )

        # ---- mocap target ----------------------------------------------
        self._mocap_body_id: int = _id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")
        self._mocap_id: int = model.body_mocapid[self._mocap_body_id]
        if self._mocap_id < 0:
            raise RuntimeError("mocap_target is not a mocap body.")

        # ---- block free-joint qpos address ----------------------------
        jnt_start: int = model.body_jntadr[self._block_body_id]
        if model.body_jntnum[self._block_body_id] == 0:
            raise RuntimeError("Block body has no joints (free-joint expected).")
        self._block_qpos_adr: int = model.jnt_qposadr[jnt_start]

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset simulation to initial state.

        Sets joint positions **and** actuator control targets so the
        PD-controller actuators (see ``panda.xml``) hold the desired
        pose rather than driving everything back to zero.
        """
        mujoco.mj_resetData(self.model, self.data)

        # Arm joints (active) — qpos + ctrl must agree
        for adr, aid, val in zip(
            self._arm_qpos_adrs, self._arm_actuator_ids, self.DEFAULT_ARM_QPOS
        ):
            self.data.qpos[adr] = val
            self.data.ctrl[aid] = val

        # Fixed joints — lock at their default values
        for adr, aid, val in zip(
            self._fixed_qpos_adrs, self._fixed_actuator_ids, self.FIXED_JOINT_VALUES
        ):
            self.data.qpos[adr] = val
            self.data.ctrl[aid] = val

        # Finger joints — qpos (ctrl stays at 0 = closed)
        for adr, val in zip(self._finger_qpos_adrs, self.DEFAULT_FINGER_QPOS):
            self.data.qpos[adr] = val
        self.data.ctrl[self._finger_actuator_id] = 0.0  # closed

        # Block — randomise xy if enabled, otherwise use default position
        adr = self._block_qpos_adr
        if self._randomize_block:
            r = self.BLOCK_RANDOM_RANGE
            rand_xy = np.random.uniform(-r, r, size=2)
            block_pos = self.DEFAULT_BLOCK_POS.copy()
            block_pos[:2] += rand_xy
            # Clamp to keep block fully on table (table half-extent 0.25×0.15, block 0.02)
            block_pos[0] = np.clip(block_pos[0], 0.22, 0.68)
            block_pos[1] = np.clip(block_pos[1], 0.03, 0.29)
        else:
            block_pos = self.DEFAULT_BLOCK_POS.copy()
        self.data.qpos[adr : adr + 3] = block_pos
        self.data.qpos[adr + 3 : adr + 7] = self.DEFAULT_BLOCK_QUAT

        # Mocap target — place at table2 top centre
        self.data.mocap_pos[self._mocap_id] = self.TARGET_POS.copy()

        mujoco.mj_forward(self.model, self.data)

        self._attach_block = False
        self._rel_pos = np.zeros(3)
        self._rel_rot = np.eye(3)

    def step(self) -> None:
        """Advance the simulation by one time step.

        When ``_attach_block`` is active the block rigidly follows the
        hand, preserving the relative pose recorded at grasp time.
        """
        # Re-assert fixed joint positions every step to prevent drift
        for adr, val in zip(self._fixed_qpos_adrs, self.FIXED_JOINT_VALUES):
            self.data.qpos[adr] = val

        mujoco.mj_step(self.model, self.data)

        # Check grasp / release conditions
        self._update_grasp_state()

        if self._attach_block:
            self._enforce_attachment()

    # ------------------------------------------------------------------
    # State properties — scene geometry
    # ------------------------------------------------------------------
    @property
    def table_position(self) -> np.ndarray:
        """World-frame position of the table body centre (x, y, z)."""
        return self.data.xpos[self._table_body_id].copy()

    @property
    def tabletop_position(self) -> np.ndarray:
        """World-frame position of the top surface centre of the table."""
        table_pos = self.data.xpos[self._table_body_id]
        half_height = self.model.geom_size[self._table_geom_id][2]  # z half-extent
        top = table_pos.copy()
        top[2] += half_height
        return top

    @property
    def table2_position(self) -> np.ndarray:
        """World-frame position of the taller table (table2) body centre."""
        return self.data.xpos[self._table2_body_id].copy()

    @property
    def table2_top_position(self) -> np.ndarray:
        """World-frame position of the top surface centre of table2 (taller)."""
        table_pos = self.data.xpos[self._table2_body_id]
        half_height = self.model.geom_size[self._table2_geom_id][2]  # z half-extent
        top = table_pos.copy()
        top[2] += half_height
        return top

    @property
    def block_position(self) -> np.ndarray:
        """World-frame position of the block centre (x, y, z)."""
        return self.data.xpos[self._block_body_id].copy()

    @property
    def block_size(self) -> np.ndarray:
        """Half-extents of the block box geom (dx, dy, dz)."""
        return self.model.geom_size[self._block_geom_id].copy()

    @property
    def target_position(self) -> np.ndarray:
        """World-frame position of the mocap target (place goal on table2)."""
        return self.data.mocap_pos[self._mocap_id].copy()

    @property
    def distance_to_target(self) -> float:
        """Euclidean distance from block centre to the mocap target."""
        return float(np.linalg.norm(self.block_position - self.target_position))

    # ------------------------------------------------------------------
    # State properties — arm
    # ------------------------------------------------------------------
    @property
    def arm_joint_positions(self) -> np.ndarray:
        """Arm joint angles [rad] for active joints (joint1, joint2, joint4, joint6)."""
        return self._read_qpos(self._arm_qpos_adrs)

    @property
    def arm_joint_velocities(self) -> np.ndarray:
        """Arm joint velocities [rad/s] for active joints (joint1, joint2, joint4, joint6)."""
        return self._read_qvel(self._arm_dof_adrs)

    @property
    def arm_joint_torques(self) -> np.ndarray:
        """Actuator forces [Nm] for active actuators (actuator1, actuator2, actuator4, actuator6)."""
        return np.array(
            [self.data.actuator_force[aid] for aid in self._arm_actuator_ids]
        )

    # ------------------------------------------------------------------
    # State properties — fingers
    # ------------------------------------------------------------------
    @property
    def finger_joint_positions(self) -> np.ndarray:
        """Finger joint positions [m] for finger_joint1 & finger_joint2."""
        return self._read_qpos(self._finger_qpos_adrs)

    # ------------------------------------------------------------------
    # State properties — end-effector
    # ------------------------------------------------------------------
    @property
    def endeffector_position(self) -> np.ndarray:
        """World-frame position of the hand (panda_hand) body centre."""
        return self.data.xpos[self._hand_body_id].copy()

    # ------------------------------------------------------------------
    # Public accessors for controller use
    # ------------------------------------------------------------------
    @property
    def hand_body_id(self) -> int:
        """MuJoCo body ID of the hand, for Jacobian computation."""
        return self._hand_body_id

    @property
    def arm_dof_adrs(self) -> List[int]:
        """DOF addresses of arm joints (columns in the Jacobian)."""
        return list(self._arm_dof_adrs)

    @property
    def arm_qpos_adrs(self) -> List[int]:
        """Qpos addresses of arm joints."""
        return list(self._arm_qpos_adrs)

    @property
    def block_body_id(self) -> int:
        """MuJoCo body ID of the block."""
        return self._block_body_id

    @property
    def block_qpos_adr(self) -> int:
        """Qpos address of the block free-joint."""
        return self._block_qpos_adr

    def set_arm_target(self, joint_positions: np.ndarray) -> None:
        """Set PD-controller targets (ctrl) for the 4 active arm actuators.

        Fixed joints (3, 5, 7) are also re-locked to their default values
        on every call, preventing drift from external forces or numerical error.
        """
        for aid, val in zip(self._arm_actuator_ids, joint_positions):
            self.data.ctrl[aid] = float(val)
        # Re-assert fixed joints every control step
        for aid, val in zip(self._fixed_actuator_ids, self.FIXED_JOINT_VALUES):
            self.data.ctrl[aid] = float(val)

    def set_gripper(self, ctrl_value: float) -> None:
        """Set the gripper actuator control signal.

        Parameters
        ----------
        ctrl_value : float
            Normalised gripper command in [0, 1] (0 = closed, 1 = open).
            Multiplied by 255 internally for the MuJoCo actuator.
        """
        self.data.ctrl[self._finger_actuator_id] = float(ctrl_value) * 255.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_qpos(self, adrs: List[int]) -> np.ndarray:
        """Read qpos at each address into a new NumPy array."""
        qpos = self.data.qpos
        return np.array([qpos[a] for a in adrs])

    def _read_qvel(self, adrs: List[int]) -> np.ndarray:
        """Read qvel at each dof address into a new NumPy array."""
        qvel = self.data.qvel
        return np.array([qvel[a] for a in adrs])

    # ------------------------------------------------------------------
    # Reliable grasping
    # ------------------------------------------------------------------

    @property
    def is_grasping(self) -> bool:
        """True when the block is rigidly attached to the hand."""
        return self._attach_block

    def _update_grasp_state(self) -> None:
        """Start grasping on finger-block contact + close cmd; release on open."""
        grip_cmd = self.data.ctrl[self._finger_actuator_id]

        if not self._attach_block:
            if self._has_finger_block_contact():
                self._attach_block = True
                self._record_grasp_pose()
        elif grip_cmd >= 200.0:
            self._attach_block = False

    def _has_finger_block_contact(self) -> bool:
        """True when both fingers are touching the block."""
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if self._block_body_id not in (b1, b2):
                continue
            if b1 == self._left_finger_body_id or b2 == self._left_finger_body_id:
                left_contact = True
            if b1 == self._right_finger_body_id or b2 == self._right_finger_body_id:
                right_contact = True
        return left_contact and right_contact

    def _record_grasp_pose(self) -> None:
        """Record the block pose relative to the hand at grasp time."""
        hand_pos = self.data.xpos[self._hand_body_id].copy()
        hand_rot = np.array(self.data.xmat[self._hand_body_id]).reshape(3, 3, order="F")
        blk_pos = self.data.xpos[self._block_body_id].copy()
        blk_rot = np.array(self.data.xmat[self._block_body_id]).reshape(3, 3, order="F")

        self._rel_pos = hand_rot.T @ (blk_pos - hand_pos)
        self._rel_rot = hand_rot.T @ blk_rot

    def _enforce_attachment(self) -> None:
        """Set the block pose to follow the hand, preserving relative pose."""
        hand_pos = self.data.xpos[self._hand_body_id].copy()
        hand_rot = np.array(self.data.xmat[self._hand_body_id]).reshape(3, 3, order="F")

        blk_pos = hand_pos + hand_rot @ self._rel_pos
        blk_rot = hand_rot @ self._rel_rot

        blk_quat = np.zeros(4)
        mujoco.mju_mat2Quat(blk_quat, blk_rot.flatten(order="F"))

        adr = self._block_qpos_adr
        self.data.qpos[adr : adr + 3] = blk_pos
        self.data.qpos[adr + 3 : adr + 7] = blk_quat
        # Zero block velocity (linear + angular)
        jnt_start: int = self.model.body_jntadr[self._block_body_id]
        dof_adr: int = self.model.jnt_dofadr[jnt_start]
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0
