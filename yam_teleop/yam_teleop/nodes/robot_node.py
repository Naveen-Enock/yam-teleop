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

from yam_teleop.hardware.yam_follower import YamFollower

# Geometry of the linear_4310 gripper, used only to draw the force-limit "hold"
# band on the torque plot. Mirrors i2rt's GripperForceLimiter: once a grasp is
# detected it regulates the holding torque to
#   tau = gripper_max_force * gripper_stroke / motor_stroke  (+ friction comp)
# See i2rt/robots/utils.py: linear_gripper_force_torque_map (motor_stroke=6.57
# rad, gripper_stroke=0.096 m) and GripperForceLimiter.update (adds ~0.3 Nm).
_GRIPPER_MOTOR_STROKE_RAD = 6.57
_GRIPPER_STROKE_M = 0.096
_GRIPPER_FRICTION_COMP_NM = 0.3
# After a gap in the command stream, cap the slew dt so a single tick can't jump
# far (velocity = delta/dt stays bounded even if dt is large).
_SLEW_DT_CAP_S = 0.05


def safe_return_to_home(left_arm: YamFollower, right_arm: YamFollower,
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


def make_yam_arm(cfg: dict, side: str) -> YamFollower:
    """Create a follower YAM arm from config (per-arm keys override top-level)."""
    arm_cfg = cfg[side]
    return YamFollower(
        channel=arm_cfg["can_channel"],
        gripper_type=arm_cfg["gripper_type"],
        zero_gravity_mode=arm_cfg["zero_gravity_mode"],
        limit_gripper_force=cfg.get("gripper_max_force", 50.0),
        gripper_torque_cap=cfg.get("gripper_torque_cap", 0.0),
        gravity_comp_factor=arm_cfg.get(
            "gravity_comp_factor", cfg.get("gravity_comp_factor")),
        kp=arm_cfg.get("kp", cfg.get("kp")),
        kd=arm_cfg.get("kd", cfg.get("kd")),
        sim=bool(cfg.get("sim", False)),
    )


def main():
    parser = argparse.ArgumentParser(description="Robot arm node")
    parser.add_argument("--config", required=True, help="Path to robot.yaml")
    parser.add_argument("--debug", action="store_true", help="Print joint states")
    parser.add_argument("--plot-gripper", action="store_true",
                        help="Open a live window graphing follower gripper "
                             "position, velocity and torque")
    parser.add_argument("--plot-window-sec", type=float, default=20.0,
                        help="Seconds of history shown in the gripper plot")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    publish_rate = cfg["publish_rate_hz"]
    state_port = cfg["zmq_state_port"]
    cmd_port = cfg["zmq_cmd_port"]
    gripper_max_open = cfg.get("gripper_max_open", 0.85)
    gripper_max_force = cfg.get("gripper_max_force", 50.0)
    # Max gripper closing/opening speed in stroke-fraction/s (0 disables). Caps
    # how fast the commanded gripper position may change, so a fast human squeeze
    # can't slam the gripper shut and spike the torque before the force limiter
    # engages. Tune with the --plot-gripper torque panel.
    gripper_max_speed = cfg.get("gripper_max_speed", 0.0)
    # Hard cap (Nm) on gripper motor torque (0 disables); see robot.yaml.
    gripper_torque_cap = cfg.get("gripper_torque_cap", 0.0)

    # Initialize both YAM arms
    sim_note = " (MuJoCo sim)" if cfg.get("sim", False) else ""
    print(f"Initializing left YAM arm{sim_note}...")
    left_arm = make_yam_arm(cfg, "left")
    print(f"Initializing right YAM arm{sim_note}...")
    right_arm = make_yam_arm(cfg, "right")

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

    # Gripper slew-rate limiter state. Seed from the actual startup gripper
    # positions so the first command slews from reality, not from a guess.
    grip_state = {
        "left": float(left_arm.get_observations()["gripper_position"][0]),
        "right": float(right_arm.get_observations()["gripper_position"][0]),
        "t": None,
    }
    if gripper_max_speed > 0:
        print(f"Gripper slew limit: {gripper_max_speed:.2f} stroke-fraction/s")
    if gripper_torque_cap > 0:
        cap_n = gripper_torque_cap * _GRIPPER_MOTOR_STROKE_RAD / _GRIPPER_STROKE_M
        print(f"Gripper torque cap: {gripper_torque_cap:.2f} Nm (~{cap_n:.0f} N gross)")

    def cmd_receiver():
        def slew_gripper(arm: str, raw_pos: float, dt: float) -> float:
            """Clamp the commanded gripper pos to <= gripper_max_speed rate."""
            target = min(raw_pos, gripper_max_open)
            prev = grip_state[arm]
            if gripper_max_speed > 0 and dt > 0:
                step = gripper_max_speed * dt
                target = float(np.clip(target, prev - step, prev + step))
            grip_state[arm] = target
            return target

        while running:
            try:
                msg = cmd_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.004)
                continue
            try:
                left_cmd = msg["left"]
                right_cmd = msg["right"]

                now = time.monotonic()
                # First command uses the dt cap so it's still rate-limited.
                dt = (_SLEW_DT_CAP_S if grip_state["t"] is None
                      else min(now - grip_state["t"], _SLEW_DT_CAP_S))
                grip_state["t"] = now

                left_target = np.array(
                    left_cmd["joint_pos"]
                    + [slew_gripper("left", left_cmd["gripper_pos"], dt)],
                    dtype=np.float32,
                )
                right_target = np.array(
                    right_cmd["joint_pos"]
                    + [slew_gripper("right", right_cmd["gripper_pos"], dt)],
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

    # Optional live gripper plot. Runs in its own process (see gripper_plot)
    # so it can't stall this loop; we only feed it a downsampled ~50Hz stream.
    plotter = None
    plot_decim = 1
    if args.plot_gripper:
        from yam_teleop.nodes.gripper_plot import GripperPlotter
        # Force-limit "hold" torque band (only meaningful for linear grippers).
        hold_band = None
        if "linear" in str(cfg["left"].get("gripper_type", "")).lower():
            tau_lo = gripper_max_force * _GRIPPER_STROKE_M / _GRIPPER_MOTOR_STROKE_RAD
            hold_band = (tau_lo, tau_lo + _GRIPPER_FRICTION_COMP_NM)
        plotter = GripperPlotter(
            window_sec=args.plot_window_sec,
            speed_limit=(gripper_max_speed if gripper_max_speed > 0 else None),
            hold_band=hold_band,
        )
        plot_decim = max(1, round(publish_rate / 50.0))
        print("Live gripper plot: position | velocity | torque (separate window)")

    rate = RateLimiter(frequency=publish_rate, warn=False)
    pbar = tqdm(desc="Robot node", unit="msg", smoothing=0.05, mininterval=1.0)
    t_debug = time.time()
    loop_i = 0

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

            # Feed the live gripper plot: position, finite-diff velocity, and
            # measured torque. Gripper is the 7th joint (index 6), so its
            # velocity is joint_velocities[6] and torque is gripper_eff.
            loop_i += 1
            if plotter is not None and loop_i % plot_decim == 0:
                plotter.push(
                    time.monotonic(),
                    msg["left"]["gripper_pos"],
                    float(left_obs["joint_velocities"][6]),
                    msg["left"]["gripper_eff"],
                    msg["right"]["gripper_pos"],
                    float(right_obs["joint_velocities"][6]),
                    msg["right"]["gripper_eff"],
                )

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
        if plotter is not None:
            plotter.close()
        state_pub.close()
        cmd_sub.close()
        ctx.term()

        # Safe shutdown: gradually move arms to home position. Guarded so the
        # motors always power down even if the move fails (e.g. one arm's
        # control loop died) — never leave the arms energized-but-stuck.
        home = np.array(cfg["home_position"], dtype=np.float32)
        max_delta = cfg.get("shutdown_max_delta", 0.01)
        try:
            safe_return_to_home(left_arm, right_arm, home, max_delta)
        except Exception as e:
            print(f"  Safe return failed ({type(e).__name__}: {e}); "
                  f"powering motors off anyway.", file=sys.stderr)

        left_arm.close()
        right_arm.close()
        print("Robot node shut down. Motors off.")


if __name__ == "__main__":
    main()
