"""Compare the LEFT vs RIGHT leader gripper-trigger signals, side by side.

The follower left gripper is proven healthy (reset opens *and* closes it), and
the leader read path is identical for both arms except the CAN channel — so if
"pressing the left trigger does nothing," the left trigger's *signal* isn't
changing. This tool prints both arms' trigger signals next to each other so you
can see exactly where they differ.

Two modes:

  --zmq   (default, NON-INVASIVE: leave teleop running)
          Subscribe to the leader node's state PUB (:zmq_port) and print the
          gripper command it is actually publishing for each arm
          (yam_leader_node publishes msg["left"|"right"]["gripper_pos"]), plus
          the teaching-handle buttons and the state-machine mode.

  --can   (INVASIVE: stop the teleop stack first)
          Read each arm's passive trigger encoder (CAN id 0x50E) directly off
          its bus — the same read LeaderArm.read() uses — and show the RAW
          signed encoder angle (rad) alongside i2rt's |raw|/0.7 normalization
          and the resulting gripper command. Raw is what reveals a mis-zeroed
          trigger that folds across raw=0. You CANNOT run this while the leader
          node is up: the trigger shares the arm's CAN bus and only one process
          can hold it.

Both modes track the running [min, max] of the gripper command per arm, so you
can squeeze+release each trigger once and compare their travel ranges. A healthy
trigger spans ~0.0 .. ~1.0; a tiny span on one arm points straight at it.

Usage (from the yam_teleop dir):
    python -m yam_teleop.scripts.compare_leader_grippers            # zmq, teleop up
    python -m yam_teleop.scripts.compare_leader_grippers --can      # teleop stopped
    python -m yam_teleop.scripts.compare_leader_grippers --can --left can_leader_l --right can_leader_r
"""

import argparse
import sys
import time

import numpy as np
import yaml

# Teaching-handle passive encoder id (i2rt get_encoder_chain -> EncoderChain([0x50E])).
ENCODER_ID = 0x50E


def _fmt_stats(stats):
    """Final travel-range summary; the interpretation key for the whole tool."""
    print("\n\nFinal gripper-command travel ranges (what the leader would send):")
    for name in ("L", "R"):
        lo, hi = stats[name]
        if lo > hi:
            print(f"  {name}: no samples")
        else:
            print(f"  {name}: cmd {lo:.2f} .. {hi:.2f}  (span {hi - lo:.2f})")
    print("A healthy trigger spans ~0.00 .. ~1.00. A tiny span on one arm = a "
          "stuck / mis-zeroed / disconnected trigger — that arm is the fault.")


def run_zmq(port, invert, rate):
    """Non-invasive: print the gripper command the leader is publishing."""
    import zmq

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)  # only the freshest state
    sub.connect(f"tcp://127.0.0.1:{port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"[zmq] subscribed to leader state on :{port} — squeeze each trigger.")
    print("(gripper_invert is applied in the leader; cmd shown is as published.)\n")

    stats = {"L": [1e9, -1e9], "R": [1e9, -1e9]}
    period = 1.0 / max(rate, 1.0)
    try:
        while True:
            try:
                msg = sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(period)
                continue
            gL = float(msg["left"]["gripper_pos"])
            gR = float(msg["right"]["gripper_pos"])
            stats["L"][0] = min(stats["L"][0], gL); stats["L"][1] = max(stats["L"][1], gL)
            stats["R"][0] = min(stats["R"][0], gR); stats["R"][1] = max(stats["R"][1], gR)
            btn = msg.get("buttons", {})
            mode = msg.get("mode", "?")
            loL, hiL = stats["L"]; loR, hiR = stats["R"]
            sys.stdout.write(
                f"\r[{mode:<12}] "
                f"L cmd={gL:.2f} [{loL:.2f},{hiL:.2f}] btn={btn.get('left')}   |   "
                f"R cmd={gR:.2f} [{loR:.2f},{hiR:.2f}] btn={btn.get('right')}   ")
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        _fmt_stats(stats)
    finally:
        sub.close()
        ctx.term()


def run_can(left_ch, right_ch, invert, rate):
    """Invasive: read the raw trigger encoder off each bus (teleop must be down)."""
    from i2rt.motor_drivers.dm_driver import CanInterface, PassiveEncoderReader

    class RawTriggerReader(PassiveEncoderReader):
        """Mirror PassiveEncoderReader.read_encoder but also return raw signed rad.

        i2rt normalizes the trigger as |raw|/0.7 (dm_driver.py), which folds about
        raw=0 — a mis-zeroed trigger can barely move `norm` even as raw swings.
        Exposing raw makes that visible.
        """

        def read_raw(self, encoder_id):
            data = [0xFF, 0x02]
            message = self.can_interface._send_message_get_response(
                encoder_id, encoder_id, data,
                expected_id=self.receive_mode.get_receive_id(0x50E), max_retry=15)
            pos_rad, vel, buttons = self._parse_encoder_message(message)
            clipped = float(np.clip(pos_rad, -0.7, 0.7))
            norm = abs(clipped) / 0.7  # == enc.position that LeaderArm.read() sees
            return pos_rad, norm, vel, buttons

    print("[can] STOP the teleop stack first — this holds the leader CAN buses.")
    print(f"[can] left={left_ch}  right={right_ch}  gripper_invert={invert}")
    print("Squeeze+release each trigger; compare raw / norm / cmd and the ranges.\n")

    readers = {}
    for name, ch in (("L", left_ch), ("R", right_ch)):
        try:
            readers[name] = RawTriggerReader(
                CanInterface(channel=ch, use_buffered_reader=False))
        except Exception as e:
            print(f"[{name}] failed to open {ch}: {type(e).__name__}: {e}")

    stats = {"L": [1e9, -1e9], "R": [1e9, -1e9]}
    period = 1.0 / max(rate, 1.0)

    def read_one(name):
        r = readers.get(name)
        if r is None:
            return None
        try:
            raw, norm, vel, btn = r.read_raw(ENCODER_ID)
            cmd = float(np.clip((1.0 - norm) if invert else norm, 0.0, 1.0))
            stats[name][0] = min(stats[name][0], cmd)
            stats[name][1] = max(stats[name][1], cmd)
            return raw, norm, cmd, btn
        except Exception as e:  # encoder didn't answer -> that's diagnostic
            return e

    try:
        while True:
            parts = []
            for name in ("L", "R"):
                res = read_one(name)
                if res is None:
                    parts.append(f"{name}: --")
                elif isinstance(res, Exception):
                    parts.append(f"{name}: NO RESPONSE ({type(res).__name__})")
                else:
                    raw, norm, cmd, btn = res
                    lo, hi = stats[name]
                    parts.append(
                        f"{name} raw={raw:+.3f} norm={norm:.2f} cmd={cmd:.2f} "
                        f"[{lo:.2f},{hi:.2f}] btn={btn}")
            sys.stdout.write("\r" + "   |   ".join(parts) + "   ")
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        _fmt_stats(stats)
    finally:
        for r in readers.values():
            iface = getattr(r, "can_interface", None)
            if iface is not None and hasattr(iface, "close"):
                try:
                    iface.close()
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser(
        description="Compare left/right leader gripper-trigger signals side by side")
    ap.add_argument("--config", default="configs/leader.yaml",
                    help="leader.yaml (for channels, port, gripper_invert)")
    ap.add_argument("--can", action="store_true",
                    help="read trigger encoders directly off CAN (stop teleop first); "
                         "default is non-invasive --zmq mode")
    ap.add_argument("--left", help="override left CAN channel (--can mode)")
    ap.add_argument("--right", help="override right CAN channel (--can mode)")
    ap.add_argument("--port", type=int, help="override leader state PUB port (--zmq mode)")
    ap.add_argument("--rate", type=float, default=20.0, help="print rate (Hz)")
    args = ap.parse_args()

    invert, port = True, 5004
    left_ch, right_ch = args.left, args.right
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        invert = bool(cfg.get("gripper_invert", True))
        port = int(cfg.get("zmq_port", 5004))
        left_ch = left_ch or cfg["left"]["can_channel"]
        right_ch = right_ch or cfg["right"]["can_channel"]
    except Exception as e:
        print(f"(could not read {args.config}: {e}; using defaults)")
        left_ch = left_ch or "can_leader_l"
        right_ch = right_ch or "can_leader_r"
    if args.port is not None:
        port = args.port

    if args.can:
        run_can(left_ch, right_ch, invert, args.rate)
    else:
        run_zmq(port, invert, args.rate)


if __name__ == "__main__":
    main()
