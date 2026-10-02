"""The few places yam_teleop reaches past i2rt's public API.

Everything here touches ``MotorChainRobot`` internals that upstream i2rt does
not expose (checked against i2rt v1.3.6). It is kept in this one module so an
i2rt upgrade only needs re-checking here, instead of patching a fork of the SDK.

All helpers are no-ops (or raise nothing) on i2rt's ``SimRobot``, which has
none of these internals.
"""

from __future__ import annotations

import numpy as np


def is_sim(robot) -> bool:
    """True for i2rt's MuJoCo SimRobot (``get_yam_robot(sim=True)``)."""
    return not hasattr(robot, "motor_chain")


def assert_alive(robot, name: str) -> None:
    """Raise if the robot's i2rt control thread or CAN motor chain has stopped.

    i2rt fails fast on lost motor comms (``enable_auto_recovery`` defaults to
    False): the motor chain stops and the robot's server thread exits. Nothing
    tells the caller, so a node would keep publishing the last (stale) state.
    Call this every tick so the node fails loudly instead.
    """
    if is_sim(robot):
        return
    thread = getattr(robot, "_server_thread", None)
    if (thread is not None and not thread.is_alive()) or not robot.motor_chain.running:
        raise RuntimeError(
            f"{name}: i2rt control loop stopped (motor comms lost?) — "
            f"check the CAN bus / arm power and restart this node")


def set_gripper_force_limit(robot, max_force_n: float) -> None:
    """Set the gripper force limit (N); <= 0 disables the limiter.

    ``get_yam_robot`` hardcodes ``limit_gripper_force=50.0``. Rebuild the
    limiter with our value instead (swapping the reference is atomic, so the
    running control thread just picks up the new one).
    """
    if is_sim(robot) or robot._gripper_index is None:
        return
    if max_force_n <= 0:
        robot._limit_gripper_force = -1.0
        return
    from i2rt.robots.utils import GripperForceLimiter

    robot._gripper_force_limiter = GripperForceLimiter(
        max_force=float(max_force_n),
        gripper_type=robot._gripper_type,
        arm_type=robot._arm_type,
        kp=float(robot._kp[robot._gripper_index]),
    )
    robot._limit_gripper_force = float(max_force_n)


def gripper_torque_cap_margin(robot, cap_nm: float) -> float | None:
    """Max |command - actual| (normalized 0..1 units) that keeps kp*error <= cap.

    The gripper is a position-PD motor, so its torque is kp * (cmd - actual) in
    motor radians. Clamping the normalized command to within this margin of the
    actual position caps that torque. Returns None when unavailable.
    """
    if is_sim(robot) or robot._gripper_index is None or cap_nm <= 0:
        return None
    kp = float(robot._kp[robot._gripper_index])
    stroke_rad = float(np.abs(robot.remapper.joint_range[0]))  # open - closed, motor rad
    if kp <= 0 or stroke_rad <= 0:
        return None
    return (cap_nm / kp) / stroke_rad


def nominal_gains(robot, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The arm's configured PD gains (first n joints); zeros on SimRobot."""
    if is_sim(robot):
        return np.zeros(n), np.zeros(n)
    return (np.asarray(robot._kp, dtype=np.float64)[:n].copy(),
            np.asarray(robot._kd, dtype=np.float64)[:n].copy())


def command_with_feedforward(robot, pos: np.ndarray, kp: np.ndarray,
                             kd: np.ndarray, torque: np.ndarray) -> None:
    """PD position command plus a per-joint feedforward torque (Nm).

    i2rt's public ``command_joint_pos`` / ``command_joint_state`` always zero
    the feedforward torque channel (``JointCommands.torques``), which the
    control loop adds on top of gravity compensation. This writes the full
    command atomically so a feedforward torque (stiction dither, virtual
    spring) can ride along. Arm-only robots (teaching-handle leaders) only.
    """
    from i2rt.robots.motor_chain_robot import JointCommands

    pos = robot._clip_robot_joint_pos_command(np.array(pos, dtype=np.float64))
    cmd = JointCommands(
        torques=np.asarray(torque, dtype=np.float64),
        pos=robot.remapper.to_robot_joint_pos_space(pos),
        vel=np.zeros(len(pos)),
        kp=np.asarray(kp, dtype=np.float64),
        kd=np.asarray(kd, dtype=np.float64),
    )
    with robot._command_lock:
        robot._commands = cmd
