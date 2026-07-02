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
  COUPLED  teleop. Feel selected by the `leader_feel` config:
           bilateral - soft PD (kp = bilateral_kp * nominal_kp, kd=0) pulling
                       the leader toward the follower's measured pose, so the
                       operator feels contact forces. bilateral_kp=0 => passive.
           damped    - stiff PD (full nominal kp/kd) chasing the leader's OWN
                       pose => viscous self-centering backdrive ("raiden feel").
           free      - zero stiffness, pure gravity-comp backdrive (gello feel).
           Entered by {"command": "couple"}.
  SUSPENDED  clutch out. kp=0, leader free; the consumer holds the follower.
             Entered by a CLUTCH button press while COUPLED.
  RESUME_MATCH  clutch in. stiff PD eases the leader back to the *follower's*
                pose on a smoothstep profile (gentle, zero velocity/jerk at both
                ends; resume_speed rad/s), then auto-advances to COUPLED. The
                follower never lurches (the consumer holds it through resume).
  (IDLE is also re-entered by {"command": "free"} / "disable_torque" — used on
   shutdown so the leader stays where it is, no pull toward the follower.)

On the node's OWN Ctrl+C / SIGTERM, both leader arms first ease to a folded safe
pose (`safe_position` in leader.yaml) under stiff PD, THEN the motors switch off
— so they settle safely instead of dropping limp. That safe pose is distinct
from the teleop start pose (the follower's home); mirrors robot_node's follower
safe return.

Teaching-handle buttons (io_inputs[0], io_inputs[1]; edge-detected; either
handle counts, i.e. global):
  - button 0 -> sub-task MARKER (published as a monotonic buttons.marker_seq
                that consumers diff — survives their CONFLATE reads).
  - button 1 -> CLUTCH toggle (COUPLED <-> SUSPENDED).

Usage:
    python -m yam_teleop.nodes.yam_leader_node --config configs/leader.yaml
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


class ClampSlew:
    """Per-tick position-clamped slew toward a target pose (both arms).

    Used in RESET: the consumer streams reset waypoints in via set_target(); each
    step() advances the setpoint toward the latest target by at most max_delta
    per joint, smoothing over any gaps between the streamed waypoints. Owns its
    own setpoint/target state (no closure / nonlocal).
    """

    def __init__(self, max_delta: float):
        self._max_delta = max_delta
        self.sp_l = self.sp_r = None
        self.target_l = self.target_r = None

    def restart(self, cur_l, cur_r) -> None:
        """Begin a fresh slew from the current pose (target = current)."""
        self.sp_l = np.asarray(cur_l, dtype=np.float64).copy()
        self.sp_r = np.asarray(cur_r, dtype=np.float64).copy()
        self.target_l = self.sp_l.copy()
        self.target_r = self.sp_r.copy()

    def set_target(self, to_l, to_r, cur_l, cur_r) -> None:
        """Update the target; seed the setpoint from current if not yet started."""
        self.target_l = np.asarray(to_l, dtype=np.float64)
        self.target_r = np.asarray(to_r, dtype=np.float64)
        if self.sp_l is None:
            self.sp_l = np.asarray(cur_l, dtype=np.float64).copy()
            self.sp_r = np.asarray(cur_r, dtype=np.float64).copy()

    def step(self):
        """Advance the setpoint one tick toward the target; return (left, right)."""
        self.sp_l = _slew(self.sp_l, self.target_l, self._max_delta)
        self.sp_r = _slew(self.sp_r, self.target_r, self._max_delta)
        return self.sp_l, self.sp_r


class SmoothSlew:
    """Time-parameterized smoothstep slew between two poses (both arms).

    Used for the clutch RESUME_MATCH return to the follower pose. s = 3a^2 - 2a^3
    has zero slope at a=0 and a=1, so velocity ramps up and down -> slow,
    graceful, no jerk. Owns its own trajectory state (no closure / nonlocal).
    """

    def __init__(self, speed: float, min_dur: float):
        self._speed = speed
        self._min_dur = min_dur
        self.from_l = self.from_r = None
        self.to_l = self.to_r = None
        self._t0 = None
        self._dur = min_dur

    def start(self, from_l, from_r, to_l, to_r) -> float:
        """Set up a slew from->to; return its duration (s)."""
        self.from_l = np.asarray(from_l, dtype=np.float64).copy()
        self.from_r = np.asarray(from_r, dtype=np.float64).copy()
        self.to_l = np.asarray(to_l, dtype=np.float64).copy()
        self.to_r = np.asarray(to_r, dtype=np.float64).copy()
        dist = max(float(np.max(np.abs(self.to_l - self.from_l))),
                   float(np.max(np.abs(self.to_r - self.from_r))))
        # Shared duration across both arms so they finish together; floored.
        self._dur = max(dist / self._speed, self._min_dur)
        self._t0 = time.monotonic()
        return self._dur

    def step(self):
        """Return (left_setpoint, right_setpoint, done) for the current time."""
        a = (1.0 if self._t0 is None
             else min((time.monotonic() - self._t0) / self._dur, 1.0))
        s = a * a * (3.0 - 2.0 * a)
        left = self.from_l + s * (self.to_l - self.from_l)
        right = self.from_r + s * (self.to_r - self.from_r)
        return left, right, a >= 1.0


def safe_return_to_pose(left: LeaderArm, right: LeaderArm,
                        safe: np.ndarray, max_delta: float) -> None:
    """Gradually drive both leader arms to a safe pose before the motors go off.

    Mirrors robot_node.safe_return_to_home: on the leader node's own Ctrl+C the
    arms should settle into a folded/parked pose under stiff PD instead of
    dropping limp. Steps along a linspace at ~60Hz so the move is slow and
    controlled. This is the parked/safe pose, NOT the teleop start pose (which
    is the follower's home, streamed in during RESET).
    """
    print("Safe shutdown: returning leader arms to safe pose...")

    left_current = left.read()[0]
    right_current = right.read()[0]

    left_distance = np.max(np.abs(left_current - safe))
    right_distance = np.max(np.abs(right_current - safe))
    max_distance = max(left_distance, right_distance)

    if max_distance < 0.05:
        print("  Already at safe pose; holding.")
        left.stiff_to(safe)
        right.stiff_to(safe)
        return

    num_steps = max(int(max_distance / max_delta), 1)
    print(f"  Moving to safe pose in {num_steps} steps "
          f"(max distance: {max_distance:.3f} rad)...")

    left_waypoints = np.linspace(left_current, safe, num_steps + 1)[1:]
    right_waypoints = np.linspace(right_current, safe, num_steps + 1)[1:]

    try:
        for i, (lw, rw) in enumerate(zip(left_waypoints, right_waypoints)):
            left.stiff_to(lw)
            right.stiff_to(rw)
            time.sleep(1.0 / 60.0)  # ~60Hz step rate
            if (i + 1) % 60 == 0:
                print(f"  {num_steps - (i + 1)} steps remaining...")
        print("  Safe pose reached.")
    except KeyboardInterrupt:
        print("  Shutdown interrupted — stopping where we are.")


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
    # Leader feel during COUPLED (active teleop). See leader.yaml for details.
    #   bilateral - soft PD toward the FOLLOWER pose (force feedback; bilateral_kp)
    #   damped    - stiff PD (full nominal kp/kd) toward the leader's OWN pose
    #               (raiden feel: viscous, self-centering backdrive)
    #   free      - zero stiffness, pure gravity-comp backdrive (gello feel)
    leader_feel = str(cfg.get("leader_feel", "bilateral")).lower()
    _valid_feels = ("bilateral", "damped", "free")
    if leader_feel not in _valid_feels:
        raise ValueError(
            f"leader_feel must be one of {_valid_feels}, got {leader_feel!r}")
    gripper_invert = bool(cfg.get("gripper_invert", True))
    match_max_delta = float(cfg.get("match_max_delta", 0.01))  # rad per tick (RESET)
    # Clutch RESUME slew: a gentle eased return to the follower pose. Speed is in
    # rad/s (the start-reset feel, ~0.4) and a smoothstep profile ramps velocity
    # from/to zero so there's no jerk; min_duration keeps even tiny moves smooth.
    resume_speed = float(cfg.get("resume_speed", 0.4))            # rad/s
    resume_min_dur = float(cfg.get("resume_min_duration", 0.5))   # s
    # Safety watchdog: if the controller stops sending its heartbeat (or any
    # command) for this long while engaged, the leader drops to IDLE (limp).
    watchdog_timeout = float(cfg.get("watchdog_timeout", 1.0))
    # Gravity-comp scale (overrides i2rt's hardcoded 1.3). Optional per-arm.
    gcf = float(cfg.get("gravity_comp_factor", 1.0))
    gcf_left = float(cfg["left"].get("gravity_comp_factor", gcf))
    gcf_right = float(cfg["right"].get("gravity_comp_factor", gcf))
    # Safe-shutdown pose: on the node's own Ctrl+C the leader arms ease (stiff
    # PD, gradual) to this pose before motors off, instead of dropping limp. 6
    # joints per arm (teaching handle has no gripper); shared by both arms. This
    # is the parked/safe pose, NOT the teleop start pose. See leader.yaml.
    safe_position = np.asarray(cfg.get("safe_position", [0.0] * 6), dtype=np.float64)
    shutdown_max_delta = float(cfg.get("shutdown_max_delta", 0.01))

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

    stop_event = threading.Event()

    def shutdown(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"YAM leader node: state on :{port} at {publish_rate}Hz, "
          f"commands on :{cmd_port}, follower SUB :{robot_state_port}, "
          f"leader_feel={leader_feel}, bilateral_kp={bilateral_kp}")
    rate = RateLimiter(frequency=publish_rate, warn=False)
    pbar = tqdm(desc="YAM leader", unit="msg", smoothing=0.05, mininterval=1.0)
    t_debug = time.time()

    # --- state machine ---
    mode = "idle"  # idle | reset | coupled | suspended | resume_match
    reset_slew = ClampSlew(match_max_delta)         # RESET: track streamed targets
    resume_slew = SmoothSlew(resume_speed, resume_min_dur)  # clutch RESUME_MATCH
    follower_left = None    # latest follower joint_pos[6]
    follower_right = None
    prev_marker_btn = False
    prev_clutch_btn = False
    # Monotonic marker counter (NOT a one-shot bool): consumers read :5004 with
    # CONFLATE at ~60Hz, so a single-message bool would be dropped. They diff
    # this counter instead — a bump that lands between reads is still seen.
    marker_seq = 0
    last_cmd_t = None  # monotonic time of the last message from a controller

    try:
        while not stop_event.is_set():
            qL, gL, bL = left.read()
            qR, gR, bR = right.read()

            # --- drain commands (PULL, non-blocking) ---
            while True:
                try:
                    cmd = cmd_pull.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                last_cmd_t = time.monotonic()  # any message proves the controller is alive
                if "command" in cmd:
                    c = cmd["command"]
                    if c == "heartbeat":
                        pass  # liveness ping only
                    elif c == "enable_torque":
                        mode = "reset"
                        # fresh slew; target = current pose until one streams in
                        reset_slew.restart(qL, qR)
                        tqdm.write("[leader] enable_torque -> RESET")
                    elif c == "couple":
                        mode = "coupled"
                        tqdm.write("[leader] couple -> COUPLED")
                    elif c in ("free", "disable_torque"):
                        # Limp / gravity-comp only (kp=0). NO pull toward the
                        # follower — used on shutdown so the leader stays put
                        # instead of snapping to the follower pose.
                        mode = "idle"
                        tqdm.write(f"[leader] {c} -> IDLE (free / gravity-comp)")
                elif "left" in cmd:
                    # Position target (reset waypoints). Only meaningful in RESET.
                    reset_slew.set_target(
                        cmd["left"]["joint_pos"], cmd["right"]["joint_pos"], qL, qR)

            # --- watchdog: controller went silent (hard crash / killed) -> limp ---
            if (last_cmd_t is not None and mode != "idle"
                    and time.monotonic() - last_cmd_t > watchdog_timeout):
                tqdm.write(f"[leader] WATCHDOG: no controller heartbeat for "
                           f">{watchdog_timeout:.1f}s -> IDLE (free)")
                mode = "idle"
                last_cmd_t = None  # disarm until a controller reconnects

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
            if marker:
                marker_seq += 1
                tqdm.write(f"[leader] MARKER #{marker_seq}")

            if clutch_edge:
                if mode == "coupled":
                    mode = "suspended"
                    tqdm.write("[leader] CLUTCH -> SUSPENDED (follower will hold)")
                elif mode == "suspended":
                    if follower_left is not None:
                        mode = "resume_match"
                        dur = resume_slew.start(
                            qL, qR, follower_left, follower_right)
                        tqdm.write(f"[leader] CLUTCH -> RESUME_MATCH "
                                   f"(easing to follower over {dur:.2f}s)")
                    else:
                        tqdm.write("[leader] CLUTCH ignored: no follower state yet")

            # --- apply control for the current mode ---
            if mode in ("idle", "suspended"):
                left.free(qL)
                right.free(qR)
            elif mode == "reset":
                # Track the consumer's externally-streamed waypoints; the slew
                # clamp just smooths over any gaps between them.
                sp_l, sp_r = reset_slew.step()
                left.stiff_to(sp_l)
                right.stiff_to(sp_r)
            elif mode == "resume_match":
                # Gentle smoothstep ease back to the follower (no external
                # streaming here, so this profile alone shapes the motion).
                sp_l, sp_r, done = resume_slew.step()
                left.stiff_to(sp_l)
                right.stiff_to(sp_r)
                if done:
                    mode = "coupled"
                    tqdm.write("[leader] RESUME_MATCH done -> COUPLED")
            elif mode == "coupled":
                if leader_feel == "damped":
                    # Raiden feel: stiff PD (nominal kp/kd) chasing the leader's
                    # OWN measured pose -> viscous, self-centering backdrive. No
                    # follower coupling, so no contact-force feedback (and no
                    # follower state needed).
                    left.stiff_to(qL)
                    right.stiff_to(qR)
                elif (leader_feel == "bilateral" and follower_left is not None
                        and bilateral_kp > 0.0):
                    # Soft bilateral PD pulling the leader toward the follower's
                    # measured pose, so the operator feels contact forces.
                    left.coupled_to(follower_left, bilateral_kp)
                    right.coupled_to(follower_right, bilateral_kp)
                else:
                    # leader_feel == "free", or bilateral with kp==0 / no
                    # follower yet: pure passive gravity-comp backdrive.
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
                    "marker_seq": marker_seq,  # monotonic; consumers diff it
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
        # Ease the leaders to the safe/parked pose before motors off (mirrors
        # robot_node). Guarded so the motors always power down even if the move
        # fails — never leave the arms energized-but-stuck.
        try:
            safe_return_to_pose(left, right, safe_position, shutdown_max_delta)
        except Exception as e:
            print(f"  Safe return failed ({type(e).__name__}: {e}); "
                  f"powering motors off anyway.", file=sys.stderr)
        left.close()
        right.close()
        print("YAM leader node shut down.")


if __name__ == "__main__":
    main()
