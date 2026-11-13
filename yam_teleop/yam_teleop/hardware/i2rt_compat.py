"""The few places yam_teleop reaches past i2rt's public API.

Everything here touches ``MotorChainRobot`` internals that upstream i2rt does
not expose (checked against i2rt v1.3.6). It is kept in this one module so an
i2rt upgrade only needs re-checking here, instead of patching a fork of the SDK.

All helpers are no-ops (or raise nothing) on i2rt's ``SimRobot``, which has
none of these internals.
"""

from __future__ import annotations

import threading

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


def patch_trigger_wrap() -> None:
    """Fold the teaching-handle trigger angle into (-pi, pi] before i2rt clips it.

    The trigger is a single-turn absolute magnetic encoder (4096 counts = 2*pi).
    Its stored zero offset can be wrong by a whole revolution -- e.g. if the
    EEPROM zero drifts or reverts on a power cycle -- which parks the resting
    trigger near +/-2*pi instead of 0. ``PassiveEncoderReader.read_encoder``
    clips to +/-range_rad (~0.7) and normalizes, assuming |pos| stays within the
    stroke, so a wrapped rest angle saturates it and freezes the gripper command
    (norm pinned to 1.0). The real stroke is far smaller than pi, so folding
    cancels any whole-revolution zero error and leaves valid readings untouched.

    Patches the class (idempotent); call before building the leader robots.
    The pre-upgrade i2rt fork carried this in dm_driver.py; upstream v1.3.6
    does not.
    """
    from i2rt.motor_drivers.dm_driver import PassiveEncoderReader

    parse = PassiveEncoderReader._parse_encoder_message
    if getattr(parse, "_yam_teleop_wrap", False):
        return

    def _parse_folded(self, message):
        pos, vel, button_state = parse(self, message)
        return (pos + np.pi) % (2 * np.pi) - np.pi, vel, button_state

    _parse_folded._yam_teleop_wrap = True
    PassiveEncoderReader._parse_encoder_message = _parse_folded


def close_robot(robot, timeout: float = 2.0) -> None:
    """``robot.close()``, but stop the CAN control thread before the bus closes.

    Upstream ``DMChainCanInterface.close()`` sets ``running = False`` and shuts
    the CAN bus straight away without joining its control thread, so a send
    already in flight hits the closed socket and the thread dies with a
    "file descriptor cannot be a negative integer (-1)" traceback after every
    clean shutdown. The pre-upgrade fork joined the thread first; do that here.
    i2rt keeps no handle to that thread, so find it by its bound target.
    """
    if is_sim(robot):
        robot.close()
        return
    # Same order as MotorChainRobot.close(): stop the server loop that feeds
    # the chain commands, then the chain loop, then let close() shut the bus.
    robot._stop_event.set()
    robot._server_thread.join()
    chain = robot.motor_chain
    chains = list(getattr(chain, "interfaces", [chain]))
    for c in chains:
        c.running = False
    for t in threading.enumerate():
        if getattr(getattr(t, "_target", None), "__self__", None) in chains:
            t.join(timeout)
    robot.close()


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
