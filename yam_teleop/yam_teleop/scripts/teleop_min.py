"""Stage-1 minimal teleop harness (no env / broker / camera / pedals).

Validates the new leader-follower core directly over ZMQ, isolating exactly the
new code (follower wrapper + yam_leader_node + bilateral + clutch) from the rest
of the stack.

Run three terminals on the robot host:
    uv run python -m yam_teleop.nodes.robot_node       --config configs/robot.yaml
    uv run python -m yam_teleop.nodes.yam_leader_node  --config configs/leader.yaml
    uv run python -m yam_teleop.scripts.teleop_min

Flow:
  1. Hold the follower at its current measured pose (so it stops floating).
  2. enable_torque -> drive the LEADER to the follower pose (the leader moves,
     the follower stays put), then disable_torque -> COUPLED.
  3. Live: stream leader joints + gripper to the follower at --rate Hz.
     Clutch (teaching-handle button 2) suspends/resumes via the leader node;
     while SUSPENDED we stop commanding so the follower holds. Marker presses
     (button 1) are just printed here (Stage 3 wires them into recording).

This talks to the SAME ZMQ contract collect_data uses, so a clean run here also
exercises the reset handshake the real pipeline depends on.
"""

import argparse
import signal
import time

import numpy as np
import zmq


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


def main():
    p = argparse.ArgumentParser(description="Stage-1 minimal leader->follower teleop")
    p.add_argument("--leader-state-port", type=int, default=5004)
    p.add_argument("--leader-cmd-port", type=int, default=5006)
    p.add_argument("--robot-state-port", type=int, default=5002)
    p.add_argument("--robot-cmd-port", type=int, default=5003)
    p.add_argument("--rate", type=float, default=60.0, help="teleop command Hz")
    p.add_argument("--match-tol", type=float, default=0.08,
                   help="rad; leader-vs-follower match tolerance for handshake")
    p.add_argument("--match-timeout", type=float, default=15.0)
    p.add_argument("--no-handshake", action="store_true",
                   help="skip enable/disable_torque; assume leader already coupled")
    args = p.parse_args()

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

    running = [True]

    def shutdown(sig, frame):
        running[0] = False

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
        # 1. Hold the follower where it is so it stops floating during the match.
        hold = _follower_cmd(leader_msg)
        hold["left"]["joint_pos"] = foll_msg["left"]["joint_pos"]
        hold["right"]["joint_pos"] = foll_msg["right"]["joint_pos"]
        hold["left"]["gripper_pos"] = foll_msg["left"]["gripper_pos"]
        hold["right"]["gripper_pos"] = foll_msg["right"]["gripper_pos"]

        input("\n*** SAFETY: clear the workspace. The LEADER arms will move to "
              "match the followers.\n    Press Enter to begin... ")

        # 2. Drive the leader to the follower pose.
        leader_cmd.send_json({"command": "enable_torque"})
        print("Matching leader -> follower...")
        deadline = time.monotonic() + args.match_timeout
        rate = 1.0 / 50.0
        while running[0] and time.monotonic() < deadline:
            foll_msg = _latest(foll_sub) or foll_msg
            robot_cmd.send_json(hold)  # keep holding the follower
            leader_cmd.send_json({
                "left": {"joint_pos": foll_msg["left"]["joint_pos"]},
                "right": {"joint_pos": foll_msg["right"]["joint_pos"]},
            })
            leader_msg = _latest(leader_sub) or leader_msg
            errL = np.max(np.abs(np.array(leader_msg["left"]["joint_pos"])
                                 - np.array(foll_msg["left"]["joint_pos"])))
            errR = np.max(np.abs(np.array(leader_msg["right"]["joint_pos"])
                                 - np.array(foll_msg["right"]["joint_pos"])))
            if max(errL, errR) < args.match_tol:
                break
            time.sleep(rate)
        print(f">> Match done (errL={errL:.3f}, errR={errR:.3f} rad).")

        # 3. Engage teleop.
        leader_cmd.send_json({"command": "disable_torque"})
        print(">> COUPLED. Live teleop started. Ctrl+C to stop.\n")

    # Live loop.
    period = 1.0 / args.rate
    suspended_prev = False
    last_print = time.time()
    while running[0]:
        t0 = time.time()
        leader_msg = _latest(leader_sub) or leader_msg
        mode = leader_msg.get("mode", "coupled")
        buttons = leader_msg.get("buttons", {})

        if buttons.get("marker"):
            print(f"[marker] sub-task boundary @ {time.strftime('%H:%M:%S')}")

        suspended = (mode == "suspended")
        if suspended != suspended_prev:
            print(f">> {'SUSPENDED (follower holding)' if suspended else 'RESUMED'}")
            suspended_prev = suspended

        if not suspended:
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
    leader_sub.close()
    foll_sub.close()
    leader_cmd.close()
    robot_cmd.close()
    ctx.term()


if __name__ == "__main__":
    main()
