"""YAM follower-arm wrapper over i2rt (replaces gello.robots.yam.YAMRobot).

The follower YAM arm is driven through i2rt's MotorChainRobot. This thin
wrapper preserves the exact observation/command surface ``robot_node`` expects:

    get_observations() -> {joint_positions, joint_velocities, joint_efforts,
                           gripper_position, timestamp_ns}
    get_joint_state() / command_joint_state() / close()

Previously this lived in the gello package (``gello.robots.yam.YAMRobot``),
which coupled the follower path to gello for no reason — that class only ever
called ``i2rt.robots.get_robot.get_yam_robot`` under the hood. Vendored here
verbatim (minus the gello base class) so the teleop stack no longer imports
gello at all.
"""

from __future__ import annotations

import time
from typing import Dict, Union

import numpy as np


class YamFollower:
    """YAM hardware wrapper with live observation refresh (i2rt-backed)."""

    def __init__(
        self,
        channel: str = "can0",
        gripper_type: Union[str, object] = "linear_4310",
        zero_gravity_mode: bool = True,
        limit_gripper_force: float = 50.0,
    ):
        # Imported lazily so importing this module never requires i2rt/CAN
        # hardware (e.g. on the dev laptop).
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        if isinstance(gripper_type, str):
            gripper_type = GripperType.from_string_name(gripper_type)

        self.robot = get_yam_robot(
            channel=channel,
            gripper_type=gripper_type,
            zero_gravity_mode=zero_gravity_mode,
            limit_gripper_force=limit_gripper_force,
        )
        self.channel = channel
        self._joint_state = np.zeros(7, dtype=np.float32)
        self._joint_velocities = np.zeros(7, dtype=np.float32)
        self._joint_efforts = np.zeros(7, dtype=np.float32)
        self._last_sample_timestamp_ns = 0
        self._last_commanded_joint_state = np.zeros(7, dtype=np.float32)

    def num_dofs(self) -> int:
        return 7

    def get_joint_state(self) -> np.ndarray:
        self._refresh_state()
        return self._joint_state.copy()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        joint_state = self._normalize_joint_state(joint_state)
        self._last_commanded_joint_state = joint_state.astype(np.float32)
        self.robot.command_joint_pos(self._last_commanded_joint_state.copy())

    def get_observations(self) -> Dict[str, np.ndarray]:
        self._refresh_state()
        return {
            "joint_positions": self._joint_state.copy(),
            "joint_velocities": self._joint_velocities.copy(),
            "joint_efforts": self._joint_efforts.copy(),
            "gripper_position": np.asarray([self._joint_state[-1]], dtype=np.float32),
            "timestamp_ns": np.asarray([self._last_sample_timestamp_ns], dtype=np.int64),
        }

    def get_joint_pos(self) -> np.ndarray:
        return self.get_joint_state()

    def command_joint_pos(self, target_pos) -> None:
        self.command_joint_state(np.asarray(target_pos, dtype=np.float32))

    def get_last_commanded_joint_state(self) -> np.ndarray:
        return self._last_commanded_joint_state.copy()

    def close(self) -> None:
        if hasattr(self.robot, "close"):
            self.robot.close()

    def _refresh_state(self) -> None:
        now_ns = time.monotonic_ns()
        joint_pos = self._normalize_joint_state(self.robot.get_joint_pos())
        if self._last_sample_timestamp_ns > 0:
            dt = max((now_ns - self._last_sample_timestamp_ns) / 1e9, 1e-6)
            self._joint_velocities = (joint_pos - self._joint_state) / dt
        else:
            self._joint_velocities = np.zeros_like(joint_pos)
        self._joint_state = joint_pos.astype(np.float32)
        self._last_sample_timestamp_ns = now_ns

        robot_obs = self.robot.get_observations()
        self._joint_efforts = robot_obs["joint_eff"][:7].copy().astype(np.float32)

    def _normalize_joint_state(self, joint_state: np.ndarray) -> np.ndarray:
        joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        if len(joint_state) > 7:
            joint_state = joint_state[:7]
        elif len(joint_state) < 7:
            joint_state = np.pad(joint_state, (0, 7 - len(joint_state)), "constant")
        return joint_state
