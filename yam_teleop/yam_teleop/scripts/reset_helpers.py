"""Shared reset helpers for the YAM leader-follower teleop scripts.

Used by both collect_yam.py (full env/recording) and teleop_min.py (raw-ZMQ
bring-up harness) so their start sequences stay identical:
  1. follower -> start/home pose
  2. leader  -> (optional waypoint) -> match the follower start pose
  3. wait for both triggers squeezed -> couple (engage teleop)
"""

import time

import numpy as np


def both_grippers_closed(leader_msg: dict, threshold: float) -> bool:
    """True when both leader triggers are squeezed past the threshold."""
    return (float(leader_msg["left"]["gripper_pos"]) < threshold
            and float(leader_msg["right"]["gripper_pos"]) < threshold)


def _stream(send, from_l, to_l, from_r, to_r, joint_speed, send_hz, label):
    """Stream interpolated 6-DOF waypoints (rate-capped) via `send(left, right)`."""
    max_d = max(np.max(np.abs(from_l - to_l)), np.max(np.abs(from_r - to_r)))
    if max_d < 1e-4:
        return
    duration = max(max_d / joint_speed, 0.05)
    n = max(int(duration * send_hz), 1)
    print(f"  reset [{label}]: {n} steps over {duration:.2f}s (max_delta={max_d:.3f} rad)")
    period = 1.0 / send_hz
    next_t = time.monotonic()
    for lw, rw in zip(np.linspace(from_l, to_l, n + 1)[1:],
                      np.linspace(from_r, to_r, n + 1)[1:]):
        send(lw, rw)
        next_t += period
        s = next_t - time.monotonic()
        if s > 0:
            time.sleep(s)
        else:
            next_t = time.monotonic()


def reset_leader(leader_cmd, get_leader_msg, home_left, home_right,
                 waypoint_left=None, waypoint_right=None,
                 gripper_threshold=0.1, joint_speed=0.4, send_hz=200.0,
                 quit_check=None):
    """Drive the leaders through an optional waypoint to the follower start pose.

    enable_torque -> (current -> waypoint -> home) -> hold until both triggers
    squeezed -> couple. The waypoint (per-arm) lets the leaders clear the
    teaching handle from the arm links before settling into the matched pose.
    `get_leader_msg()` returns the latest leader state dict (or None);
    `quit_check()` (optional) returning True aborts with SystemExit.
    """
    def send(lw, rw):
        leader_cmd.send_json({
            "left": {"joint_pos": np.asarray(lw).tolist(), "gripper_pos": 0.0},
            "right": {"joint_pos": np.asarray(rw).tolist(), "gripper_pos": 0.0},
        })

    leader_cmd.send_json({"command": "enable_torque", "gripper_limp": True})

    msg = None
    deadline = time.monotonic() + 2.0
    while msg is None and time.monotonic() < deadline:
        msg = get_leader_msg()
        time.sleep(0.005)
    if msg is None:
        raise RuntimeError("No leader messages received after enable_torque")

    cur_l = np.asarray(msg["left"]["joint_pos"], dtype=float)
    cur_r = np.asarray(msg["right"]["joint_pos"], dtype=float)
    home_l = np.asarray(home_left, dtype=float)
    home_r = np.asarray(home_right, dtype=float)

    if waypoint_left is not None and waypoint_right is not None:
        wp_l = np.asarray(waypoint_left, dtype=float)
        wp_r = np.asarray(waypoint_right, dtype=float)
        _stream(send, cur_l, wp_l, cur_r, wp_r, joint_speed, send_hz, "leader->waypoint")
        _stream(send, wp_l, home_l, wp_r, home_r, joint_speed, send_hz, "leader->home")
    else:
        _stream(send, cur_l, home_l, cur_r, home_r, joint_speed, send_hz, "leader->home")

    print("  Leader at start pose. Squeeze both triggers to start...")
    while True:
        send(home_l, home_r)  # hold at home while waiting
        msg = get_leader_msg()
        if msg is not None and both_grippers_closed(msg, gripper_threshold):
            break
        time.sleep(0.005)
        if quit_check is not None and quit_check():
            raise SystemExit("reset-quit")

    leader_cmd.send_json({"command": "couple"})


def reset_follower_raw(robot_cmd, get_follower_msg, home_full,
                       joint_speed=0.4, send_hz=100.0):
    """Interpolate the follower to the home pose over raw ZMQ (no env).

    Used by teleop_min, which doesn't run the env/broker. `home_full` is the
    14-dim [left6, lgrip, right6, rgrip] pose; grippers are commanded to their
    home values. collect_yam uses env.reset() for this instead.
    """
    msg = None
    deadline = time.monotonic() + 5.0
    while msg is None and time.monotonic() < deadline:
        msg = get_follower_msg()
        time.sleep(0.01)
    if msg is None:
        raise RuntimeError("No follower state received for reset")

    cur_l = np.asarray(msg["left"]["joint_pos"], dtype=float)
    cur_r = np.asarray(msg["right"]["joint_pos"], dtype=float)
    home = np.asarray(home_full, dtype=float)
    home_l, lgrip = home[:6], float(home[6])
    home_r, rgrip = home[7:13], float(home[13])

    def send(lw, rw):
        robot_cmd.send_json({
            "left": {"joint_pos": np.asarray(lw).tolist(), "gripper_pos": lgrip},
            "right": {"joint_pos": np.asarray(rw).tolist(), "gripper_pos": rgrip},
        })

    _stream(send, cur_l, home_l, cur_r, home_r, joint_speed, send_hz, "follower->home")
    send(home_l, home_r)  # final hold (robot_node CONFLATEs and keeps re-applying)
