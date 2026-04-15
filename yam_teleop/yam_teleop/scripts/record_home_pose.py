"""Record the current YAM arm pose and save it as the home_position.

Workflow:
  1. Launch robot_node only (zero_gravity_mode: true makes arms limp).
  2. Run this script.
  3. Physically move both arms to the desired reset pose.
  4. Press ENTER to capture. Confirm with y/N.
  5. env.yaml's home_position is rewritten in place (comments preserved).

Both YAM and GELLO use this pose at the start of each collect_data run
(GELLO joints are calibrated to the same joint space as YAM, so the same
target works for both).

Usage:
    python -m yam_teleop.scripts.record_home_pose \
        --env-config configs/env.yaml
"""

import argparse
import re
import select
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import zmq


def fmt_deg(joints):
    return " ".join(f"{np.degrees(j):+7.2f}" for j in joints)


def main():
    parser = argparse.ArgumentParser(description="Record YAM home pose")
    parser.add_argument("--env-config", required=True,
                        help="Path to env.yaml")
    parser.add_argument("--robot-state-port", type=int, default=5002,
                        help="robot_node state port (default: 5002)")
    parser.add_argument("--stale-warn-ms", type=int, default=500,
                        help="Warn if state is older than this (default: 500)")
    parser.add_argument("--round-deg", type=float, default=1.0,
                        help="Snap captured joints to this step in degrees "
                             "(default: 1.0 = whole degrees, 0 = no rounding)")
    args = parser.parse_args()

    env_path = Path(args.env_config)
    if not env_path.exists():
        print(f"ERROR: env config not found: {env_path}", file=sys.stderr)
        sys.exit(1)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(f"tcp://127.0.0.1:{args.robot_state_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"Subscribed to robot state on tcp://127.0.0.1:"
          f"{args.robot_state_port}")
    print("Waiting for first state message... "
          "(robot_node must be running)")

    t0 = time.time()
    last_msg = None
    while last_msg is None:
        try:
            last_msg = sub.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            if time.time() - t0 > 5.0:
                print("  ...still waiting. Is robot_node up?")
                t0 = time.time()
            time.sleep(0.05)

    print()
    print("Arms must be in zero-gravity mode (robot.yaml: "
          "zero_gravity_mode: true) — they should be limp now.")
    print("Position both arms to the desired reset pose, then press ENTER.")
    print("Ctrl-C to abort.")
    print()

    last_print = 0.0
    captured = None

    try:
        while True:
            try:
                last_msg = sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            now = time.time()
            if now - last_print > 0.1:
                age_ms = (time.time_ns()
                          - last_msg["timestamp_ns"]) / 1e6
                stale = age_ms > args.stale_warn_ms
                tag = " [STALE]" if stale else ""
                left = last_msg["left"]["joint_pos"]
                right = last_msg["right"]["joint_pos"]
                sys.stdout.write(
                    f"\r  L (deg): {fmt_deg(left)} |"
                    f" R (deg): {fmt_deg(right)}{tag}   "
                )
                sys.stdout.flush()
                last_print = now

            r, _, _ = select.select([sys.stdin], [], [], 0.02)
            if r:
                _ = sys.stdin.readline()
                captured = last_msg
                break

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    finally:
        sub.close()
        ctx.term()

    raw_left = list(captured["left"]["joint_pos"])
    raw_right = list(captured["right"]["joint_pos"])

    def snap(v_rad, step_deg):
        if step_deg <= 0:
            return v_rad
        return np.radians(round(np.degrees(v_rad) / step_deg) * step_deg)

    left = [snap(v, args.round_deg) for v in raw_left]
    right = [snap(v, args.round_deg) for v in raw_right]

    print()
    print()
    print("Captured pose (raw):")
    print(f"  Left  (deg):  "
          + ", ".join(f"{np.degrees(v):+7.2f}" for v in raw_left))
    print(f"  Right (deg):  "
          + ", ".join(f"{np.degrees(v):+7.2f}" for v in raw_right))

    if args.round_deg > 0:
        print(f"\nSnapped to {args.round_deg}° steps "
              "(--round-deg 0 to keep raw):")
        print(f"  Left  (deg):  "
              + ", ".join(f"{np.degrees(v):+7.2f}" for v in left))
        print(f"  Right (deg):  "
              + ", ".join(f"{np.degrees(v):+7.2f}" for v in right))

    print(f"\n  Left  (rad):  "
          + ", ".join(f"{v:+.4f}" for v in left))
    print(f"  Right (rad):  "
          + ", ".join(f"{v:+.4f}" for v in right))
    print()

    answer = input(f"Write to {env_path}? [y/N] ").strip().lower()
    if answer != "y":
        print("Not modified.")
        sys.exit(0)

    # Preserve gripper entries from the existing config — reset closes
    # grippers as its final phase regardless of these values.
    with open(env_path) as f:
        env_cfg = yaml.safe_load(f)
    old_home = env_cfg.get("home_position", [0.0] * 14)
    if not isinstance(old_home, list) or len(old_home) != 14:
        old_home = [0.0] * 14
    left_grip = old_home[6]
    right_grip = old_home[13]

    new_home = left + [left_grip] + right + [right_grip]

    def fmt(vals):
        return ", ".join(f"{v:.6f}" for v in vals)

    new_block = (
        "[" + fmt(new_home[:7]) + ",\n"
        "                " + fmt(new_home[7:]) + "]"
    )

    content = env_path.read_text()
    new_content, n = re.subn(
        r"home_position:\s*\[[^\]]*\]",
        f"home_position: {new_block}",
        content,
    )
    if n != 1:
        print(f"ERROR: expected exactly 1 home_position match in "
              f"{env_path}, found {n}. File not modified.",
              file=sys.stderr)
        sys.exit(1)

    env_path.write_text(new_content)
    print(f"Updated home_position in {env_path}")
    print("GELLO and YAM will both reset to this pose on next collect_data run.")


if __name__ == "__main__":
    main()
