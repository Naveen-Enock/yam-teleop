"""Robot node: publishes YAM arm state via ZMQ PUB at 200Hz and receives joint commands.

Usage:
    python -m yam_teleop.nodes.robot_node --config configs/robot.yaml
"""

import argparse
import signal
import sys
import threading
import time

from loop_rate_limiters import RateLimiter
import numpy as np
from tqdm import tqdm
import yaml
import zmq

from gello.robots.yam import YAMRobot


def safe_return_to_home(left_arm: YAMRobot, right_arm: YAMRobot,
                        home: np.ndarray, max_delta: float) -> None:
    """Gradually move both arms to home position before shutdown."""
    print("Safe shutdown: returning arms to home position...")

    left_current = left_arm.get_joint_state()
    right_current = right_arm.get_joint_state()

    left_distance = np.max(np.abs(left_current - home))
    right_distance = np.max(np.abs(right_current - home))
    max_distance = max(left_distance, right_distance)

    if max_distance < 0.05:
        print("  Already at home.")
        return

    num_steps = max(int(max_distance / max_delta), 1)
    print(f"  Moving to home in {num_steps} steps "
          f"(max distance: {max_distance:.3f} rad)...")

    left_waypoints = np.linspace(left_current, home, num_steps + 1)[1:]
    right_waypoints = np.linspace(right_current, home, num_steps + 1)[1:]

    try:
        for i, (lw, rw) in enumerate(zip(left_waypoints, right_waypoints)):
            left_arm.command_joint_state(lw)
            right_arm.command_joint_state(rw)
            time.sleep(1.0 / 60.0)  # ~60Hz step rate

            if (i + 1) % 60 == 0:
                remaining = num_steps - (i + 1)
                print(f"  {remaining} steps remaining...")

        print("  Home position reached.")
    except KeyboardInterrupt:
        print("  Shutdown interrupted — stopping where we are.")


def make_yam_arm(arm_cfg: dict, gripper_max_force: float) -> YAMRobot:
    """Create a YAMRobot from config."""
    return YAMRobot(
        channel=arm_cfg["can_channel"],
        gripper_type=arm_cfg["gripper_type"],
        zero_gravity_mode=arm_cfg["zero_gravity_mode"],
        limit_gripper_force=gripper_max_force,
    )


def main():
    parser = argparse.ArgumentParser(description="Robot arm node")
    parser.add_argument("--config", required=True, help="Path to robot.yaml")
    parser.add_argument("--debug", action="store_true", help="Print joint states")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    publish_rate = cfg["publish_rate_hz"]
    state_port = cfg["zmq_state_port"]
    cmd_port = cfg["zmq_cmd_port"]
    gripper_max_open = cfg.get("gripper_max_open", 0.85)
    gripper_max_force = cfg.get("gripper_max_force", 50.0)

    # Initialize both YAM arms
    print("Initializing left YAM arm...")
    left_arm = make_yam_arm(cfg["left"], gripper_max_force)
    print("Initializing right YAM arm...")
    right_arm = make_yam_arm(cfg["right"], gripper_max_force)

    # ZMQ sockets
    ctx = zmq.Context()

    state_pub = ctx.socket(zmq.PUB)
    state_pub.setsockopt(zmq.SNDHWM, 1)
    state_pub.bind(f"tcp://127.0.0.1:{state_port}")

    cmd_sub = ctx.socket(zmq.SUB)
    cmd_sub.setsockopt(zmq.CONFLATE, 1)  # keep only latest command
    cmd_sub.connect(f"tcp://127.0.0.1:{cmd_port}")
    cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    time.sleep(0.5)

    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Command receiver thread — applies commands as fast as they arrive.
    # Any uncaught exception here would silently kill the daemon thread,
    # leaving YAM frozen on its last command while the i2rt control loop
    # keeps re-applying it. Catch broadly and log so we never lose teleop.
    cmd_count = [0]
    cmd_errors = [0]

    def cmd_receiver():
        while running:
            try:
                msg = cmd_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.004)
                continue
            try:
                left_cmd = msg["left"]
                right_cmd = msg["right"]

                left_target = np.array(
                    left_cmd["joint_pos"] + [min(left_cmd["gripper_pos"], gripper_max_open)],
                    dtype=np.float32,
                )
                right_target = np.array(
                    right_cmd["joint_pos"] + [min(right_cmd["gripper_pos"], gripper_max_open)],
                    dtype=np.float32,
                )

                left_arm.command_joint_state(left_target)
                right_arm.command_joint_state(right_target)
                cmd_count[0] += 1
            except Exception as e:
                cmd_errors[0] += 1
                tqdm.write(f"[robot_node] cmd_receiver error #{cmd_errors[0]}: "
                           f"{type(e).__name__}: {e}")
            time.sleep(0.004)  # ~250Hz poll (matches i2rt control loop)

    cmd_thread = threading.Thread(target=cmd_receiver, daemon=True)
    cmd_thread.start()

    print(f"Robot node: state on :{state_port} at {publish_rate}Hz, "
          f"commands on :{cmd_port}")
    rate = RateLimiter(frequency=publish_rate, warn=False)
    pbar = tqdm(desc="Robot node", unit="msg", smoothing=0.05, mininterval=1.0)
    t_debug = time.time()

    try:
        while running:
            timestamp_ns = time.time_ns()

            # Read state from both arms
            left_obs = left_arm.get_observations()
            right_obs = right_arm.get_observations()

            msg = {
                "timestamp_ns": timestamp_ns,
                "left": {
                    "joint_pos": left_obs["joint_positions"][:6].tolist(),
                    "joint_vel": left_obs["joint_velocities"][:6].tolist(),
                    "joint_eff": left_obs["joint_efforts"][:6].tolist(),
                    "gripper_pos": float(left_obs["gripper_position"][0]),
                    "gripper_eff": float(left_obs["joint_efforts"][6]),
                },
                "right": {
                    "joint_pos": right_obs["joint_positions"][:6].tolist(),
                    "joint_vel": right_obs["joint_velocities"][:6].tolist(),
                    "joint_eff": right_obs["joint_efforts"][:6].tolist(),
                    "gripper_pos": float(right_obs["gripper_position"][0]),
                    "gripper_eff": float(right_obs["joint_efforts"][6]),
                },
            }
            state_pub.send_json(msg)

            pbar.update(1)
            pbar.set_postfix_str(
                f"cmds_applied={cmd_count[0]} errors={cmd_errors[0]}")
            if args.debug and time.time() - t_debug >= 5.0:
                np.set_printoptions(precision=3, suppress=True)
                lp = np.array(msg["left"]["joint_pos"])
                rp = np.array(msg["right"]["joint_pos"])
                tqdm.write(
                    f"  L joints: {lp}  grip: {msg['left']['gripper_pos']:.3f}\n"
                    f"  R joints: {rp}  grip: {msg['right']['gripper_pos']:.3f}")
                t_debug = time.time()

            rate.sleep()

    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
    finally:
        pbar.close()
        state_pub.close()
        cmd_sub.close()
        ctx.term()

        # Safe shutdown: gradually move arms to home position
        home = np.array(cfg["home_position"], dtype=np.float32)
        max_delta = cfg.get("shutdown_max_delta", 0.01)
        safe_return_to_home(left_arm, right_arm, home, max_delta)

        left_arm.close()
        right_arm.close()
        print("Robot node shut down. Motors off.")


if __name__ == "__main__":
    main()
