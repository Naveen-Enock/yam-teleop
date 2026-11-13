"""YAM follower-arm wrapper over i2rt (replaces gello.robots.yam.YAMRobot).

The follower YAM arm is driven through i2rt's ``get_yam_robot``. This thin
wrapper preserves the exact observation/command surface ``robot_node`` expects:

    get_observations() -> {joint_positions, joint_velocities, joint_efforts,
                           gripper_position, timestamp_ns}
    get_joint_state() / command_joint_state() / close()

Joint vectors are always 7 long: 6 arm joints + the normalized gripper
(0 = closed, 1 = open).

Built on upstream i2rt with no SDK patches; the two things upstream doesn't
offer (a configurable gripper force limit, a gripper torque cap) are done here
via ``i2rt_compat``. ``sim=True`` swaps in i2rt's MuJoCo SimRobot so the whole
stack can run without CAN hardware.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Union

import numpy as np

from yam_teleop.hardware import i2rt_compat, per_joint

class YamFollower:
    """YAM hardware wrapper with live observation refresh (i2rt-backed)."""

    def __init__(
        self,
        channel: str = "can0",
        gripper_type: Union[str, object] = "linear_4310",
        zero_gravity_mode: bool = True,
        limit_gripper_force: float = 50.0,
        gripper_torque_cap: float = -1.0,
        gravity_comp_factor: Union[None, float, Sequence[float]] = None,
        kp: Optional[Sequence[float]] = None,
        kd: Optional[Sequence[float]] = None,
        sim: bool = False,
    ):
        # Imported lazily so importing this module never requires i2rt/CAN
        # hardware (e.g. on the dev laptop).
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        if isinstance(gripper_type, str):
            gripper_type = GripperType.from_string_name(gripper_type)

        self.robot = get_yam_robot(
            channel=channel,
            arm_type=ArmType.YAM,
            gripper_type=gripper_type,
            zero_gravity_mode=zero_gravity_mode,
            gravity_comp_factor=per_joint(gravity_comp_factor, 6, "gravity_comp_factor"),  # None -> i2rt default
            sim=sim,
        )
        self.channel = channel
        self.name = f"follower[{'sim' if sim else channel}]"

        i2rt_compat.set_gripper_force_limit(self.robot, limit_gripper_force)
        if kp is not None or kd is not None:
            # Arm gains only; keep i2rt's gripper gains (7th entry).
            cur_kp, cur_kd = i2rt_compat.nominal_gains(self.robot, 7)
            new_kp = cur_kp.copy() if kp is None else np.append(per_joint(kp, 6, "kp"), cur_kp[6])
            new_kd = cur_kd.copy() if kd is None else np.append(per_joint(kd, 6, "kd"), cur_kd[6])
            if not i2rt_compat.is_sim(self.robot):
                self.robot.update_kp_kd(new_kp, new_kd)
        self._torque_cap_margin = i2rt_compat.gripper_torque_cap_margin(
            self.robot, gripper_torque_cap)
        # kd/kp of the gripper PD: converts measured velocity into the position
        # offset that cancels the damping term (see _cap_gripper_command).
        self._gripper_kd_over_kp = i2rt_compat.gripper_kd_over_kp(self.robot)

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
        joint_state = self._normalize_joint_state(joint_state).copy()
        if self._torque_cap_margin is not None:
            joint_state[6] = self._cap_gripper_command(joint_state[6])
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
            i2rt_compat.close_robot(self.robot)

    def _cap_gripper_command(self, cmd: float) -> float:
        """Cap the gripper's net motor torque, always on (no stall detection).

        The motor applies tau = kp*(cmd - actual) - kd*vel (the command's
        velocity target is 0). Clamping only the spring term would also cap
        free-motion speed, since the spring has to beat the damping. So clamp
        the command around the point where the two cancel,
        actual + (kd/kp)*vel, which bounds |tau| <= gripper_torque_cap at every
        command: in free motion the spring may lead by whatever offsets damping,
        and the instant contact stops the gripper (vel -> 0) the allowance
        collapses to cap/kp. Uses a fresh read so the clamp sees the current
        velocity, not the last publish-loop sample.
        """
        obs = self.robot.get_observations()
        actual = float(obs["gripper_pos"][0])
        vel = float(obs["gripper_vel"][0])
        center = actual + self._gripper_kd_over_kp * vel
        m = self._torque_cap_margin
        return float(np.clip(cmd, center - m, center + m))

    def _refresh_state(self) -> None:
        i2rt_compat.assert_alive(self.robot, self.name)
        now_ns = time.monotonic_ns()
        robot_obs = self.robot.get_observations()
        joint_pos = self._normalize_joint_state(np.concatenate(
            [robot_obs["joint_pos"], robot_obs["gripper_pos"]]))
        if self._last_sample_timestamp_ns > 0:
            dt = max((now_ns - self._last_sample_timestamp_ns) / 1e9, 1e-6)
            self._joint_velocities = (joint_pos - self._joint_state) / dt
        else:
            self._joint_velocities = np.zeros_like(joint_pos)
        self._joint_state = joint_pos.astype(np.float32)
        self._last_sample_timestamp_ns = now_ns

        # i2rt reports arm efforts (6) and the gripper effort separately.
        self._joint_efforts = np.concatenate(
            [robot_obs["joint_eff"], robot_obs["gripper_eff"]]).astype(np.float32)

    def _normalize_joint_state(self, joint_state: np.ndarray) -> np.ndarray:
        joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        if len(joint_state) > 7:
            joint_state = joint_state[:7]
        elif len(joint_state) < 7:
            joint_state = np.pad(joint_state, (0, 7 - len(joint_state)), "constant")
        return joint_state
