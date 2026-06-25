"""YAM leader-arm node: command-driven server (replaces gello_node).

Two YAM arms fitted with teaching handles act as leaders for leader-follower
teleoperation. Each arm runs in i2rt gravity-compensation mode so the operator
backdrives it; the teaching handle provides a trigger (read as the gripper
command) and two buttons.

Wire contract is IDENTICAL to the old gello_node, so env.py / collect_data.py /
replay_episode.py / sync_broker.py are unchanged:
  - zmq_port (5004): state PUB at publish_rate_hz. Message adds two *additive*
    fields ("mode", "buttons") that old consumers ignore.
  - zmq_cmd_port (5006): commands PULL (position targets + torque control).

Bilateral force feedback subscribes to the follower state stream published by
robot_node (robot_state_port, 5002).

State machine (mode field in the published message):
  IDLE     kp=0, gravity-comp only. Initial state; operator moves leader freely.
  RESET    stiff PD; slews to streamed position targets. Entered by
           {"command": "enable_torque"} (the reset cycle drives the leader to
           the follower's start pose). torque_enabled=True here (mirrors gello).
  COUPLED  teleop. Soft bilateral PD: kp = bilateral_kp * nominal_kp pulling the
           leader toward the follower's measured pose, so the operator feels
           contact forces. Entered by {"command": "disable_torque"}.
           bilateral_kp=0 => pure passive backdrive (== old gello feel).
  SUSPENDED  clutch out. kp=0, leader free; the consumer holds the follower.
             Entered by a CLUTCH button press while COUPLED.
  RESUME_MATCH  clutch in. stiff PD slews the leader to the *follower's* pose
                (follower never lurches), then auto-advances to COUPLED.

Teaching-handle buttons (io_inputs[0], io_inputs[1]; edge-detected; either
handle counts, i.e. global):
  - button 0 -> one-shot sub-task MARKER (published as buttons.marker=True for
                exactly one message).
  - button 1 -> CLUTCH toggle (COUPLED <-> SUSPENDED).

Usage:
    python -m yam_teleop.nodes.yam_leader_node --config configs/leader.yaml
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


class LeaderArm:
    """One YAM leader arm + its teaching-handle encoder (trigger + buttons)."""

    def __init__(self, channel: str, gripper_invert: bool = True,
                 gravity_comp_factor: float = 1.0):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        self.robot = get_yam_robot(
            channel=channel,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            zero_gravity_mode=True,
        )
        # i2rt's get_yam_robot hardcodes gravity_comp_factor=1.3 — a 30%
        # over-compensation tuned for a position-controlled FOLLOWER (the PD
        # masks it and it helps overcome friction). A leader backdrives with
        # kp=0, so 1.3x makes the arm actively float/rise ("shoots up"). It's a
        # plain per-tick scalar attribute, so override it here (no i2rt edit).
        # 1.0 = neutral buoyancy; lower (~0.9) if it still drifts up, raise if
        # it sags.
        self.robot.gravity_comp_factor = float(gravity_comp_factor)
        self.motor_chain = self.robot.motor_chain
        self.nominal_kp = np.asarray(self.robot._kp, dtype=np.float64).copy()
        self.nominal_kd = np.asarray(self.robot._kd, dtype=np.float64).copy()
        self.n = len(self.nominal_kp)  # 6 — a teaching handle has no gripper motor
        self._zero = np.zeros(self.n)
        self._gripper_invert = gripper_invert

    def read(self):
        """Return (joint_pos[n], gripper_pos in 0..1, buttons[2] bools)."""
        obs = self.robot.get_observations()
        qpos = np.asarray(obs["joint_pos"], dtype=np.float64)[: self.n]

        gripper_pos = 1.0
        buttons = [False, False]
        states = self.motor_chain.get_same_bus_device_states()
        if states:  # None until the encoder thread has read at least once
            enc = states[0]
            g = float(enc.position)  # i2rt already normalizes to ~0..1
            gripper_pos = (1.0 - g) if self._gripper_invert else g
            gripper_pos = float(np.clip(gripper_pos, 0.0, 1.0))
            io = list(enc.io_inputs)
            buttons = [bool(io[0]) if len(io) > 0 else False,
                       bool(io[1]) if len(io) > 1 else False]
        return qpos, gripper_pos, buttons

    def free(self, qpos: np.ndarray) -> None:
        """Gravity-comp only: zero PD, command current pose so kp=0 takes hold."""
        self.robot.update_kp_kd(self._zero.copy(), self._zero.copy())
        self.robot.command_joint_pos(qpos[: self.n])

    def stiff_to(self, setpoint: np.ndarray) -> None:
        """Stiff PD toward a setpoint (match / reset drive)."""
        self.robot.update_kp_kd(self.nominal_kp.copy(), self.nominal_kd.copy())
        self.robot.command_joint_pos(np.asarray(setpoint, dtype=np.float64)[: self.n])

    def coupled_to(self, follower_pos: np.ndarray, bilateral_kp: float) -> None:
        """Soft bilateral PD pulling the leader toward the follower pose."""
        self.robot.update_kp_kd(self.nominal_kp * bilateral_kp, self._zero.copy())
        self.robot.command_joint_pos(np.asarray(follower_pos, dtype=np.float64)[: self.n])

    def close(self) -> None:
        self.robot.close()


def _slew(setpoint: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    """Step setpoint toward target by at most max_delta per joint."""
    delta = np.clip(target - setpoint, -max_delta, max_delta)
    return setpoint + delta


def main():
    parser = argparse.ArgumentParser(description="YAM leader arm node")
    parser.add_argument("--config", required=True, help="Path to leader.yaml")
    parser.add_argument("--debug", action="store_true", help="Print joint values")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    publish_rate = cfg["publish_rate_hz"]
    port = cfg["zmq_port"]
    cmd_port = cfg.get("zmq_cmd_port", 5006)
    robot_state_port = cfg.get("robot_state_port", 5002)
    bilateral_kp = float(cfg.get("bilateral_kp", 0.0))
    gripper_invert = bool(cfg.get("gripper_invert", True))
    match_max_delta = float(cfg.get("match_max_delta", 0.01))  # rad per tick
    match_tol = float(cfg.get("match_tol", 0.05))              # rad
    # Gravity-comp scale (overrides i2rt's hardcoded 1.3). Optional per-arm.
    gcf = float(cfg.get("gravity_comp_factor", 1.0))
    gcf_left = float(cfg["left"].get("gravity_comp_factor", gcf))
    gcf_right = float(cfg["right"].get("gravity_comp_factor", gcf))

    print(f"Initializing left YAM leader arm (gravity_comp_factor={gcf_left})...")
    left = LeaderArm(cfg["left"]["can_channel"], gripper_invert, gcf_left)
    print(f"Initializing right YAM leader arm (gravity_comp_factor={gcf_right})...")
    right = LeaderArm(cfg["right"]["can_channel"], gripper_invert, gcf_right)

    # ZMQ sockets
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.bind(f"tcp://127.0.0.1:{port}")

    cmd_pull = ctx.socket(zmq.PULL)
    cmd_pull.bind(f"tcp://127.0.0.1:{cmd_port}")

    # Follower state stream (for bilateral feedback). Direct SUB, CONFLATE — same
    # "bypass the broker for freshness" pattern env.get_latest_gello() uses.
    foll_sub = ctx.socket(zmq.SUB)
    foll_sub.setsockopt(zmq.CONFLATE, 1)
    foll_sub.connect(f"tcp://127.0.0.1:{robot_state_port}")
    foll_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    time.sleep(0.5)

    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"YAM leader node: state on :{port} at {publish_rate}Hz, "
          f"commands on :{cmd_port}, follower SUB :{robot_state_port}, "
          f"bilateral_kp={bilateral_kp}")
    rate = RateLimiter(frequency=publish_rate, warn=False)
    pbar = tqdm(desc="YAM leader", unit="msg", smoothing=0.05, mininterval=1.0)
    t_debug = time.time()

    # --- state machine ---
    mode = "idle"  # idle | reset | coupled | suspended | resume_match
    target_left = None    # streamed position targets (reset)
    target_right = None
    setpoint_left = None   # slewed setpoint during reset / resume_match
    setpoint_right = None
    follower_left = None    # latest follower joint_pos[6]
    follower_right = None
    prev_marker_btn = False
    prev_clutch_btn = False

    def enter_match(to_left, to_right, cur_left, cur_right):
        nonlocal setpoint_left, setpoint_right, target_left, target_right
        target_left = np.asarray(to_left, dtype=np.float64)
        target_right = np.asarray(to_right, dtype=np.float64)
        setpoint_left = np.asarray(cur_left, dtype=np.float64).copy()
        setpoint_right = np.asarray(cur_right, dtype=np.float64).copy()

    try:
        while running:
            qL, gL, bL = left.read()
            qR, gR, bR = right.read()

            # --- drain commands (PULL, non-blocking) ---
            while True:
                try:
                    cmd = cmd_pull.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if "command" in cmd:
                    if cmd["command"] == "enable_torque":
                        mode = "reset"
                        # default target = current pose until a target streams in
                        enter_match(qL, qR, qL, qR)
                        tqdm.write("[leader] enable_torque -> RESET")
                    elif cmd["command"] == "disable_torque":
                        mode = "coupled"
                        tqdm.write("[leader] disable_torque -> COUPLED")
                elif "left" in cmd:
                    # Position target (reset waypoints). Only meaningful in RESET.
                    tl = np.asarray(cmd["left"]["joint_pos"], dtype=np.float64)
                    tr = np.asarray(cmd["right"]["joint_pos"], dtype=np.float64)
                    target_left, target_right = tl, tr
                    if setpoint_left is None:
                        setpoint_left, setpoint_right = qL.copy(), qR.copy()

            # --- latest follower state (bilateral / resume target) ---
            try:
                fmsg = foll_sub.recv_json(flags=zmq.NOBLOCK)
                follower_left = np.asarray(fmsg["left"]["joint_pos"], dtype=np.float64)
                follower_right = np.asarray(fmsg["right"]["joint_pos"], dtype=np.float64)
            except zmq.Again:
                pass

            # --- buttons (edge-detect; either handle) ---
            marker_btn = bL[0] or bR[0]
            clutch_btn = bL[1] or bR[1]
            marker = marker_btn and not prev_marker_btn
            clutch_edge = clutch_btn and not prev_clutch_btn
            prev_marker_btn = marker_btn
            prev_clutch_btn = clutch_btn

            if clutch_edge:
                if mode == "coupled":
                    mode = "suspended"
                    tqdm.write("[leader] CLUTCH -> SUSPENDED (follower will hold)")
                elif mode == "suspended":
                    if follower_left is not None:
                        mode = "resume_match"
                        enter_match(follower_left, follower_right, qL, qR)
                        tqdm.write("[leader] CLUTCH -> RESUME_MATCH (slewing to follower)")
                    else:
                        tqdm.write("[leader] CLUTCH ignored: no follower state yet")

            # --- apply control for the current mode ---
            if mode in ("idle", "suspended"):
                left.free(qL)
                right.free(qR)
            elif mode in ("reset", "resume_match"):
                setpoint_left = _slew(setpoint_left, target_left, match_max_delta)
                setpoint_right = _slew(setpoint_right, target_right, match_max_delta)
                left.stiff_to(setpoint_left)
                right.stiff_to(setpoint_right)
                if mode == "resume_match":
                    err = max(float(np.max(np.abs(qL - target_left))),
                              float(np.max(np.abs(qR - target_right))))
                    if err < match_tol:
                        mode = "coupled"
                        tqdm.write("[leader] RESUME_MATCH done -> COUPLED")
            elif mode == "coupled":
                if follower_left is not None and bilateral_kp > 0.0:
                    left.coupled_to(follower_left, bilateral_kp)
                    right.coupled_to(follower_right, bilateral_kp)
                else:
                    # bilateral_kp == 0 (pure passive) or no follower yet:
                    # behave like a free leader.
                    left.free(qL)
                    right.free(qR)

            # --- publish (always; identical shape to gello_node + extras) ---
            msg = {
                "timestamp_ns": time.time_ns(),
                "left": {"joint_pos": qL.tolist(), "gripper_pos": gL},
                "right": {"joint_pos": qR.tolist(), "gripper_pos": gR},
                "torque_enabled": (mode == "reset"),
                "mode": mode,
                "buttons": {
                    "marker": bool(marker),
                    "clutch": mode == "suspended",
                    "left": [bool(bL[0]), bool(bL[1])],
                    "right": [bool(bR[0]), bool(bR[1])],
                },
            }
            pub.send_json(msg)

            pbar.update(1)
            pbar.set_postfix_str(f"mode={mode} gripL={gL:.2f} gripR={gR:.2f}")
            if args.debug and time.time() - t_debug >= 5.0:
                np.set_printoptions(precision=3, suppress=True)
                tqdm.write(f"  [{mode}] L {qL} g{gL:.2f} btn{bL}\n"
                           f"          R {qR} g{gR:.2f} btn{bR}")
                t_debug = time.time()

            rate.sleep()

    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        raise
    finally:
        pbar.close()
        pub.close()
        cmd_pull.close()
        foll_sub.close()
        ctx.term()
        left.close()
        right.close()
        print("YAM leader node shut down.")


if __name__ == "__main__":
    main()
