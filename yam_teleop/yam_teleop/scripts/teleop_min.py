"""Stage-1 minimal teleop harness (no env / broker / camera / pedals).

Validates the new leader-follower core directly over ZMQ, isolating exactly the
new code (follower wrapper + yam_leader_node + bilateral + clutch) from the rest
of the stack.

Run three terminals on the robot host:
    uv run python -m yam_teleop.nodes.robot_node       --config configs/robot.yaml
    uv run python -m yam_teleop.nodes.yam_leader_node  --config configs/leader.yaml
    uv run python -m yam_teleop.scripts.teleop_min

Flow:
  1. Follower -> start pose (raw ZMQ; collect_yam uses env.reset()).
  2. enable_torque -> drive the LEADER through the waypoint to the follower
     pose; squeeze both triggers -> couple -> COUPLED.
  3. Soft handoff: ease the follower from its reset pose onto the live leader
     stream over ~0.5s so engaging can't lurch it.
  4. Live: stream leader joints + gripper to the follower at --rate Hz.
     Clutch (handle button 1) suspends/resumes via the leader node; while
     SUSPENDED/RESUME_MATCH we stop commanding so the follower holds. Marker
     presses (button 0) are just printed here (collect_yam records them).

This talks to the SAME ZMQ contract collect_data uses, so a clean run here also
exercises the reset handshake the real pipeline depends on.
"""

import argparse
import signal
import threading
import time

import yaml
import zmq

from yam_teleop.scripts.reset_helpers import reset_follower_raw, reset_leader


def start_leader_heartbeat(port: int, hz: float = 10.0):
    """Background daemon pinging the leader node so it knows we're alive.

    If this process dies hard, the thread dies with it and the leader's watchdog
    drops it to IDLE. Own thread + ZMQ context (sockets aren't thread-safe).
    """
    stop = threading.Event()

    def loop():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{port}")
        period = 1.0 / hz
        while not stop.is_set():
            try:
                sock.send_json({"command": "heartbeat"})
            except Exception:
                break
            stop.wait(period)
        sock.close()
        ctx.term()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop, t


def _latest(sub, timeout_s=0.0):
    """Non-blocking (or briefly blocking) read of the freshest CONFLATE msg."""
    deadline = time.monotonic() + timeout_s
    msg = None
    while True:
        try:
            msg = sub.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            if msg is not None or time.monotonic() >= deadline:
                return msg
            time.sleep(0.002)


def _follower_cmd(leader_msg):
    """Build a robot_node command from a leader state message."""
    return {
        "left": {
            "joint_pos": leader_msg["left"]["joint_pos"],
            "gripper_pos": leader_msg["left"]["gripper_pos"],
        },
        "right": {
            "joint_pos": leader_msg["right"]["joint_pos"],
            "gripper_pos": leader_msg["right"]["gripper_pos"],
        },
    }


def _blend(a, b, s):
    """Element-wise (1-s)*a + s*b for two equal-length numeric lists."""
    return [(1.0 - s) * float(x) + s * float(y) for x, y in zip(a, b)]


def main():
    p = argparse.ArgumentParser(description="Stage-1 minimal leader->follower teleop")
    p.add_argument("--leader-state-port", type=int, default=5004)
    p.add_argument("--leader-cmd-port", type=int, default=5006)
    p.add_argument("--robot-state-port", type=int, default=5002)
    p.add_argument("--robot-cmd-port", type=int, default=5003)
    p.add_argument("--rate", type=float, default=60.0, help="teleop command Hz")
    p.add_argument("--env-config", default="yam_teleop/configs/env.yaml",
                   help="env.yaml — read for home_position + leader_reset_waypoint")
    p.add_argument("--no-handshake", action="store_true",
                   help="skip the reset; assume leader already coupled")
    args = p.parse_args()
    period = 1.0 / args.rate  # teleop command period (used in handoff + live loop)

    ctx = zmq.Context()
    leader_sub = ctx.socket(zmq.SUB)
    leader_sub.setsockopt(zmq.CONFLATE, 1)
    leader_sub.connect(f"tcp://127.0.0.1:{args.leader_state_port}")
    leader_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    foll_sub = ctx.socket(zmq.SUB)
    foll_sub.setsockopt(zmq.CONFLATE, 1)
    foll_sub.connect(f"tcp://127.0.0.1:{args.robot_state_port}")
    foll_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    leader_cmd = ctx.socket(zmq.PUSH)
    leader_cmd.setsockopt(zmq.LINGER, 0)
    leader_cmd.connect(f"tcp://127.0.0.1:{args.leader_cmd_port}")

    robot_cmd = ctx.socket(zmq.PUB)
    robot_cmd.setsockopt(zmq.SNDHWM, 1)
    robot_cmd.bind(f"tcp://127.0.0.1:{args.robot_cmd_port}")

    # Liveness heartbeat -> leader watchdog frees the arm if we die hard.
    hb_stop, hb_thread = start_leader_heartbeat(args.leader_cmd_port)

    stop = threading.Event()

    def shutdown(sig, frame):
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Waiting for first leader + follower messages...")
    leader_msg = _latest(leader_sub, timeout_s=10.0)
    foll_msg = _latest(foll_sub, timeout_s=10.0)
    if leader_msg is None or foll_msg is None:
        raise RuntimeError("No leader/follower state. Are robot_node and "
                           "yam_leader_node running?")
    print(">> Both streams alive.")

    if not args.no_handshake:
        env_cfg = yaml.safe_load(open(args.env_config))
        home = env_cfg["home_position"]
        waypoint = env_cfg.get("leader_reset_waypoint")

        input("\n*** SAFETY: clear the workspace. Followers move to the start "
              "pose, then the LEADERS move (waypoint -> match).\n"
              "    Press Enter to begin... ")

        # Same start sequence as collect_yam (but follower homed over raw ZMQ
        # since teleop_min has no env/broker).
        # 1. Follower -> start pose.
        print("Resetting follower to start pose...")
        reset_follower_raw(robot_cmd, lambda: _latest(foll_sub), home)

        # 2. Leader -> waypoint -> match follower; squeeze both triggers; couple.
        print("Resetting leader (waypoint -> match; squeeze both triggers)...")
        reset_leader(
            leader_cmd, lambda: _latest(leader_sub),
            home_left=home[:6], home_right=home[7:13],
            waypoint_left=waypoint, waypoint_right=waypoint,
            quit_check=lambda: stop.is_set(),
        )
        leader_msg = _latest(leader_sub) or leader_msg

        # 3. Soft handoff. Squeezing the triggers to start displaces the leader
        # off the matched pose; commanding the follower straight to that pose in
        # one 60Hz step makes it lurch (the noise heard on bring-up). Ramp a
        # blend 0->1 over HANDOFF_S so the follower eases from its reset pose
        # onto the live leader stream. (collect_yam avoids this via its broker
        # warm-up gap before the first env.step.)
        foll_msg = _latest(foll_sub) or foll_msg
        f_l, f_r = foll_msg["left"]["joint_pos"], foll_msg["right"]["joint_pos"]
        f_lg = float(foll_msg["left"]["gripper_pos"])
        f_rg = float(foll_msg["right"]["gripper_pos"])
        handoff_s, t_h = 0.5, time.monotonic()
        while not stop.is_set():
            a = (time.monotonic() - t_h) / handoff_s
            if a >= 1.0:
                break
            s = a * a * (3.0 - 2.0 * a)  # smoothstep ease-in/out
            leader_msg = _latest(leader_sub) or leader_msg
            l, r = leader_msg["left"], leader_msg["right"]
            robot_cmd.send_json({
                "left": {"joint_pos": _blend(f_l, l["joint_pos"], s),
                         "gripper_pos": (1.0 - s) * f_lg + s * float(l["gripper_pos"])},
                "right": {"joint_pos": _blend(f_r, r["joint_pos"], s),
                          "gripper_pos": (1.0 - s) * f_rg + s * float(r["gripper_pos"])},
            })
            time.sleep(period)
        print(">> COUPLED + handed off. Live teleop started. Ctrl+C to stop.\n")

    # Live loop.
    paused_prev = False
    last_marker_seq = leader_msg.get("buttons", {}).get("marker_seq", 0)
    last_print = time.time()
    while not stop.is_set():
        t0 = time.time()
        leader_msg = _latest(leader_sub) or leader_msg
        mode = leader_msg.get("mode", "coupled")
        buttons = leader_msg.get("buttons", {})

        marker_seq = buttons.get("marker_seq", 0)
        if marker_seq != last_marker_seq:
            print(f"[marker] #{marker_seq} sub-task boundary @ {time.strftime('%H:%M:%S')}")
            last_marker_seq = marker_seq

        # Pause through both "suspended" and "resume_match" so the follower
        # holds still until the leader has fully re-matched (mode == "coupled").
        paused = mode in ("suspended", "resume_match")
        if paused != paused_prev:
            print(f">> {'PAUSED (follower holding)' if paused else 'RESUMED'}")
            paused_prev = paused

        if not paused:
            robot_cmd.send_json(_follower_cmd(leader_msg))

        if time.time() - last_print >= 1.0:
            gl = leader_msg["left"]["gripper_pos"]
            gr = leader_msg["right"]["gripper_pos"]
            print(f"  mode={mode}  gripL={gl:.2f} gripR={gr:.2f}", end="\r")
            last_print = time.time()

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)

    print("\nteleop_min stopped. (robot_node handles its own safe return on Ctrl+C.)")
    hb_stop.set()
    # Leave the leader limp (gravity-comp), NOT coupled — otherwise it would
    # keep pulling toward the follower after we exit.
    try:
        leader_cmd.send_json({"command": "free"})
        time.sleep(0.1)
    except Exception:
        pass
    hb_thread.join(timeout=1.0)
    leader_sub.close()
    foll_sub.close()
    leader_cmd.close()
    robot_cmd.close()
    ctx.term()


if __name__ == "__main__":
    main()
