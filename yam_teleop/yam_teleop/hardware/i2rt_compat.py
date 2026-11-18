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


def install_grasp_controller(robot, max_force_n: float, torque_cap_nm: float,
                             max_speed: float = 0.0):
    """Replace i2rt's gripper force limiter with our GripperGraspController.

    i2rt calls ``robot._gripper_force_limiter.update(gripper_state)`` every
    control tick (~1 kHz) whenever ``robot._limit_gripper_force > 0``, and
    uses the returned raw position as the gripper command. We swap in our own
    controller there (the reference swap is atomic, so the running control
    thread just picks it up). ``get_yam_robot`` hardcodes a 50 N limiter.

    The grip torque for ``max_force_n`` comes from i2rt's own per-gripper
    force->torque map (linear or crank) plus i2rt's friction comp, so
    ``gripper_max_force`` means what it did with i2rt's limiter.
    Returns the controller, or None when disabled (max_force_n <= 0) / sim.
    """
    if is_sim(robot) or robot._gripper_index is None:
        return None
    if max_force_n <= 0:
        robot._limit_gripper_force = -1.0
        return None
    from i2rt.robots.utils import GripperForceLimiter

    from yam_teleop.hardware.gripper_grasp import (
        FRICTION_COMP_NM, GripperGraspController)

    gi = robot._gripper_index
    kp = float(robot._kp[gi])
    # Borrow i2rt's force->torque map for this gripper type.
    ref = GripperForceLimiter(
        max_force=float(max_force_n),
        gripper_type=robot._gripper_type,
        arm_type=robot._arm_type,
        kp=kp,
    )
    force_map = ref.gripper_force_torque_map

    def hold_torque(raw_pos: float) -> float:
        return abs(float(force_map(current_angle=raw_pos))) + FRICTION_COMP_NM

    lo, hi = (float(x) for x in robot.remapper.joint_limits[0])
    controller = GripperGraspController(
        kp=kp,
        kd=float(robot._kd[gi]),
        raw_closed=lo,
        raw_open=hi,
        hold_torque_fn=hold_torque,
        torque_cap=float(torque_cap_nm),
        max_speed=float(max_speed),
    )
    robot._gripper_force_limiter = controller
    robot._limit_gripper_force = float(max_force_n)
    return controller


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
