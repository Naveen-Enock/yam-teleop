"""GELLO leader arm node: command-driven server.

Publishes joint positions at 200Hz via ZMQ PUB.
Accepts position commands and torque control via ZMQ PULL.

Port layout:
  - zmq_port (5004): state PUB (always publishing at 200Hz)
  - zmq_cmd_port (5006): commands PULL (position targets + torque control)

Usage:
    python -m yam_teleop.nodes.gello_node --config configs/gello.yaml
"""

import argparse
import signal
import sys
import time

from loop_rate_limiters import RateLimiter
import numpy as np
from tqdm import tqdm
import yaml
import zmq

from gello.dynamixel.driver import (
    ADDR_TORQUE_ENABLE,
    EXTENDED_POSITION_CONTROL_MODE,
    TORQUE_DISABLE,
)
from gello.robots.dynamixel import DynamixelRobot


def make_gello_arm(arm_cfg: dict, baudrate: int) -> DynamixelRobot:
    """Create a DynamixelRobot for one GELLO arm from config."""
    gripper_cfg = arm_cfg["gripper_config"]
    return DynamixelRobot(
        joint_ids=arm_cfg["joint_ids"],
        joint_offsets=arm_cfg["joint_offsets"],
        joint_signs=arm_cfg["joint_signs"],
        real=True,
        port=arm_cfg["port"],
        baudrate=baudrate,
        gripper_config=tuple(gripper_cfg),
        start_joints=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    )


ADDR_GOAL_POSITION = 116


def _arm_joint_to_raw(robot, arm_joints: np.ndarray) -> np.ndarray:
    """Convert arm joint-space values to raw servo angles (excludes gripper).

    Inverse of DynamixelRobot.get_joint_state() for arm joints only:
        raw = joint * sign + offset
    """
    n = len(arm_joints)
    return arm_joints * robot._joint_signs[:n] + robot._joint_offsets[:n]


def _write_arm_positions(arm: DynamixelRobot, arm_ids: list,
                         joint_target: np.ndarray) -> None:
    """Write position commands to arm servos."""
    driver = arm._driver
    raw = _arm_joint_to_raw(arm, joint_target)
    with driver._lock:
        for dxl_id, angle in zip(arm_ids, raw):
            position_value = int(angle * 2048 / np.pi)
            driver._packetHandler.write4ByteTxRx(
                driver._portHandler, dxl_id, ADDR_GOAL_POSITION, position_value
            )


def _setup_arm_for_position_control(arm: DynamixelRobot,
                                    gripper_limp: bool = True) -> None:
    """Switch one arm to extended position control mode with torque enabled.

    Mode 4 (extended position control) supports multi-turn: the goal register
    accepts any int32, so commands that would resolve to negative counts (or
    counts > 4095) work even when the encoder is parked near a wrap boundary.
    Mode 3 (single-turn) silently fails on out-of-range goals — joints
    calibrated near the 0/2π seam end up unreachable in one direction.

    Recalibrates joint offsets after mode change to prevent wrapping:
    counts can shift relative to offsets calibrated in a different mode.
    """
    driver = arm._driver
    gripper_id = driver._ids[-1]
    n_arm = len(driver._ids) - 1  # exclude gripper

    # Read position BEFORE mode change
    arm._last_pos = None
    pos_before = arm.get_joint_state()[:n_arm].copy()
    arm._last_pos = None

    # Switch to extended position control mode
    driver.set_torque_mode(False)
    arm._torque_on = False
    driver.set_operating_mode(EXTENDED_POSITION_CONTROL_MODE)
    driver.set_torque_mode(True)
    arm._torque_on = True

    # Wait for background reading thread to pick up post-mode-change values
    # (thread reads at ~1kHz; 50ms guarantees multiple fresh reads)
    time.sleep(0.05)

    # Read position AFTER mode change
    arm._last_pos = None
    pos_after = arm.get_joint_state()[:n_arm].copy()
    arm._last_pos = None

    # Fix offsets if mode change caused wrapping (> 90° jump on any joint)
    for i in range(n_arm):
        diff = pos_after[i] - pos_before[i]
        tqdm.write(f"  [GELLO] Joint {i}: before={np.degrees(pos_before[i]):.1f} "
                   f"after={np.degrees(pos_after[i]):.1f} diff={np.degrees(diff):.1f} deg")
        if abs(diff) > np.pi / 2:
            correction = np.round(diff / (2 * np.pi)) * 2 * np.pi
            arm._joint_offsets[i] += correction * arm._joint_signs[i]
            tqdm.write(f"    -> offset corrected by {np.degrees(correction):.0f} deg")

    # Final clean read with corrected offsets
    arm._last_pos = None
    arm.get_joint_state()

    # Disable torque on gripper servo only (leave it limp)
    if gripper_limp:
        with driver._lock:
            driver._packetHandler.write1ByteTxRx(
                driver._portHandler, gripper_id, ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE,
            )


def _disable_arm_torque(arm: DynamixelRobot) -> None:
    """Disable torque on one arm and flush serial port.

    Does NOT clear _last_pos — physical position hasn't changed,
    and clearing would cause the smoothing filter to reinitialize
    with a potentially wrapped reading.
    """
    try:
        arm._driver.set_torque_mode(False)
    except Exception:
        pass
    arm._torque_on = False
    with arm._driver._lock:
        arm._driver._portHandler.clearPort()


def main():
    parser = argparse.ArgumentParser(description="GELLO leader arm node")
    parser.add_argument("--config", required=True, help="Path to gello.yaml")
    parser.add_argument("--debug", action="store_true",
                        help="Print joint positions")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    baudrate = cfg["baudrate"]
    publish_rate = cfg["publish_rate_hz"]
    port = cfg["zmq_port"]
    cmd_port = cfg.get("zmq_cmd_port", 5006)

    # Initialize both GELLO arms
    print("Initializing left GELLO arm...")
    left_arm = make_gello_arm(cfg["left"], baudrate)
    print("Initializing right GELLO arm...")
    right_arm = make_gello_arm(cfg["right"], baudrate)

    # Arm servo IDs (excluding gripper) for position writing
    left_arm_ids = list(left_arm._driver._ids[:-1])
    right_arm_ids = list(right_arm._driver._ids[:-1])

    # ZMQ sockets
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.bind(f"tcp://127.0.0.1:{port}")

    cmd_pull = ctx.socket(zmq.PULL)
    cmd_pull.bind(f"tcp://127.0.0.1:{cmd_port}")

    time.sleep(0.5)

    # Graceful shutdown
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"GELLO node: state on :{port} at {publish_rate}Hz, "
          f"commands on :{cmd_port}")
    rate = RateLimiter(frequency=publish_rate, warn=False)
    pbar = tqdm(desc="GELLO node", unit="msg", smoothing=0.05, mininterval=1.0)
    t_debug = time.time()

    # Command state
    torque_enabled = False
    current_target_left = None
    current_target_right = None

    try:
        while running:
            # 1. Drain all pending commands (PULL, non-blocking)
            while True:
                try:
                    cmd = cmd_pull.recv_json(flags=zmq.NOBLOCK)
                    if "command" in cmd:
                        # Control command: enable_torque / disable_torque
                        if cmd["command"] == "enable_torque":
                            gripper_limp = cmd.get("gripper_limp", True)
                            tqdm.write("[GELLO] LEFT arm torque enable:")
                            _setup_arm_for_position_control(
                                left_arm, gripper_limp)
                            tqdm.write("[GELLO] RIGHT arm torque enable:")
                            _setup_arm_for_position_control(
                                right_arm, gripper_limp)
                            torque_enabled = True
                            tqdm.write(f"Torque ENABLED "
                                       f"(gripper_limp={gripper_limp})")
                        elif cmd["command"] == "disable_torque":
                            _disable_arm_torque(left_arm)
                            _disable_arm_torque(right_arm)
                            torque_enabled = False
                            current_target_left = None
                            current_target_right = None
                            tqdm.write("Torque DISABLED")
                    elif "left" in cmd:
                        # Position command (same format as YAM robot_node)
                        current_target_left = np.array(
                            cmd["left"]["joint_pos"])
                        current_target_right = np.array(
                            cmd["right"]["joint_pos"])
                except zmq.Again:
                    break

            # 2. Apply position if torque enabled
            if torque_enabled and current_target_left is not None:
                _write_arm_positions(left_arm, left_arm_ids,
                                     current_target_left)
                _write_arm_positions(right_arm, right_arm_ids,
                                     current_target_right)

            # 3. Read and publish (ALWAYS — never skip)
            timestamp_ns = time.time_ns()
            left_state = left_arm.get_joint_state()
            right_state = right_arm.get_joint_state()
            msg = {
                "timestamp_ns": timestamp_ns,
                "left": {
                    "joint_pos": left_state[:6].tolist(),
                    "gripper_pos": float(left_state[6]),
                },
                "right": {
                    "joint_pos": right_state[:6].tolist(),
                    "gripper_pos": float(right_state[6]),
                },
                "torque_enabled": torque_enabled,
            }
            pub.send_json(msg)

            # 4. Rate display
            torque_str = "ON" if torque_enabled else "off"
            pbar.set_postfix_str(
                f"torque={torque_str}  "
                f"wrist L={left_state[5]:.3f} R={right_state[5]:.3f}")
            pbar.update(1)
            if args.debug and time.time() - t_debug >= 5.0:
                np.set_printoptions(precision=3, suppress=True)
                tqdm.write(
                    f"  L joints: {left_state[:6]}  "
                    f"grip: {left_state[6]:.3f}\n"
                    f"  R joints: {right_state[:6]}  "
                    f"grip: {right_state[6]:.3f}")
                t_debug = time.time()

            # 5. Rate limiting
            rate.sleep()

    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        pbar.close()
        if torque_enabled:
            _disable_arm_torque(left_arm)
            _disable_arm_torque(right_arm)
        pub.close()
        cmd_pull.close()
        ctx.term()
        print("GELLO node shut down.")


if __name__ == "__main__":
    main()
