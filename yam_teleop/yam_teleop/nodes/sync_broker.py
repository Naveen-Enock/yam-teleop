"""Sync broker: subscribes to camera, robot, and GELLO nodes, publishes synchronized
observations at 60Hz (camera-driven).

Design:
- Dedicated receiver threads continuously fill ring buffers for robot and GELLO state
- Main thread blocks on camera frames (60Hz)
- On each camera frame, interpolates robot/GELLO state at the camera timestamp
- Publishes a unified multipart message

Usage:
    python -m yam_teleop.nodes.sync_broker --config configs/broker.yaml
"""

import argparse
import collections
import signal
import sys
import threading
import time

from tqdm import tqdm
import yaml
import zmq


class TimestampedBuffer:
    """Thread-safe ring buffer of timestamped JSON messages.

    Samples are stored in chronological order. Supports lookup by timestamp
    with linear interpolation between the two bracketing samples.
    """

    def __init__(self, name: str, maxlen: int = 200,
                 nearest_fields: set | None = None):
        self.name = name
        self.nearest_fields = nearest_fields or set()
        self._buffer = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, msg: dict) -> None:
        with self._lock:
            self._buffer.append(msg)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def interpolate_at(self, target_ns: int) -> tuple[dict | None, float]:
        """Interpolate state at target_ns.

        Returns (interpolated_state, age_ms).
        age_ms is how far target_ns is beyond the latest sample (0 if within range).
        Returns (None, _) if buffer has fewer than 2 samples.
        """
        with self._lock:
            n = len(self._buffer)
            if n == 0:
                return None, float("inf")
            if n == 1:
                msg = self._buffer[0]
                age_ms = (target_ns - msg["timestamp_ns"]) / 1e6
                return msg, age_ms

            earliest_ns = self._buffer[0]["timestamp_ns"]
            latest_ns = self._buffer[-1]["timestamp_ns"]

            # Target is after all samples — return latest, report staleness
            if target_ns >= latest_ns:
                age_ms = (target_ns - latest_ns) / 1e6
                return self._buffer[-1], age_ms

            # Target is before all samples — return earliest
            if target_ns <= earliest_ns:
                return self._buffer[0], 0.0

            # Find the two bracketing samples
            # Search from the end (most recent) since target is usually near the end
            for i in range(n - 1, 0, -1):
                t_after = self._buffer[i]["timestamp_ns"]
                t_before = self._buffer[i - 1]["timestamp_ns"]
                if t_before <= target_ns <= t_after:
                    before = self._buffer[i - 1]
                    after = self._buffer[i]
                    alpha = (target_ns - t_before) / (t_after - t_before)
                    result = _interpolate_state(
                        before, after, alpha, target_ns, self.nearest_fields)
                    return result, 0.0

            # Shouldn't reach here, but fallback to latest
            return self._buffer[-1], 0.0


def _interpolate_state(before: dict, after: dict, alpha: float,
                       target_ns: int,
                       nearest_fields: set = frozenset()) -> dict:
    """Interpolate between two state messages.

    Interpolates all numeric fields under "left" and "right" sub-dicts.
    Fields in nearest_fields use nearest-neighbor; others use linear interpolation.
    """
    result = {"timestamp_ns": target_ns}
    for side in ("left", "right"):
        result[side] = {}
        for key in before[side]:
            v0 = before[side][key]
            v1 = after[side][key]
            if key in nearest_fields:
                result[side][key] = v0 if alpha < 0.5 else v1
            elif isinstance(v0, list):
                result[side][key] = [
                    v0[j] + alpha * (v1[j] - v0[j]) for j in range(len(v0))
                ]
            else:
                # scalar (e.g. gripper_pos)
                result[side][key] = v0 + alpha * (v1 - v0)
    # Pass through non-interpolatable fields from the later sample
    result["torque_enabled"] = after.get("torque_enabled", False)
    return result


def receiver_thread(socket: zmq.Socket, buffer: TimestampedBuffer,
                    running: threading.Event) -> None:
    """Continuously receive JSON messages and append to buffer."""
    while not running.is_set():
        try:
            msg = socket.recv_json(flags=zmq.NOBLOCK)
            buffer.append(msg)
        except zmq.Again:
            time.sleep(0.001)  # brief sleep to avoid busy-spin


def main():
    parser = argparse.ArgumentParser(description="Sync broker")
    parser.add_argument("--config", required=True, help="Path to broker.yaml")
    parser.add_argument("--debug", action="store_true", help="Print interpolation details")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # `camera_meta_port` is the metadata-only PUB on camera_node — small
    # JSON, no image bytes. The image stream goes directly from camera_node
    # to clients (e.g. env.py) without passing through the broker.
    camera_meta_port = cfg["camera_meta_port"]
    robot_port = cfg["robot_port"]
    publish_port = cfg["publish_port"]
    staleness_ms = cfg["staleness_threshold_ms"]

    ctx = zmq.Context()

    # Camera-metadata subscriber — we block on this in the main thread
    cam_sub = ctx.socket(zmq.SUB)
    cam_sub.setsockopt(zmq.RCVHWM, 1)
    cam_sub.connect(f"tcp://127.0.0.1:{camera_meta_port}")
    cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Robot subscriber — received in background thread
    robot_sub = ctx.socket(zmq.SUB)
    robot_sub.setsockopt(zmq.RCVHWM, 10)
    robot_sub.connect(f"tcp://127.0.0.1:{robot_port}")
    robot_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Publisher
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.bind(f"tcp://127.0.0.1:{publish_port}")

    time.sleep(1.0)

    # Per-field interpolation config (default: linear for all fields)
    interp_cfg = cfg.get("interpolation", {})
    robot_nearest = set(interp_cfg.get("robot", {}).get("nearest", []))

    # Ring buffer for robot state. At 200Hz, 200 samples = 1 second of history.
    # GELLO state is recorded directly from gello_node by clients (see env.py
    # get_latest_gello), bypassing the broker entirely.
    robot_buffer = TimestampedBuffer("robot", maxlen=200,
                                     nearest_fields=robot_nearest)

    # Start receiver thread
    stop_event = threading.Event()

    robot_thread = threading.Thread(
        target=receiver_thread, args=(robot_sub, robot_buffer, stop_event),
        daemon=True,
    )
    robot_thread.start()

    # Signal handling
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Sync broker: cam_meta:{camera_meta_port} robot:{robot_port} "
          f"-> pub:{publish_port}")
    print(f"Staleness threshold: {staleness_ms}ms")
    print("Waiting for streams...")

    pbar = tqdm(desc="Broker", unit="fr", smoothing=0.05, mininterval=1.0)

    # Rolling 1-second timing accumulators (microseconds). Reported via
    # tqdm.write so it appears alongside the rate without clobbering pbar.
    t_recv_us = 0.0
    t_interp_us = 0.0
    t_send_us = 0.0
    t_total_us = 0.0
    timing_count = 0
    last_timing_report = time.monotonic()

    try:
        while running:
            t_iter_start = time.perf_counter()

            # --- Block on camera-metadata frame (drives 60Hz rate) ---
            # Just JSON — image bytes flow directly from camera_node to
            # downstream clients without passing through the broker.
            t0 = time.perf_counter()
            cam_meta = cam_sub.recv_json()
            cam_timestamp_ns = cam_meta["timestamp_ns"]
            camera_names = cam_meta["cameras"]
            t_recv = time.perf_counter() - t0

            # --- Interpolate robot at camera timestamp ---
            t0 = time.perf_counter()
            robot_state, robot_age_ms = robot_buffer.interpolate_at(cam_timestamp_ns)
            t_interp = time.perf_counter() - t0

            # Check if robot stream is missing or stale
            if robot_state is None or robot_age_ms > staleness_ms:
                reason = ("no data" if robot_state is None
                          else f"{robot_age_ms:.1f}ms")
                tqdm.write(f"HALTED — stale robot stream ({reason}). "
                           "Waiting for recovery...")
                # Drain camera-metadata frames without publishing until recovery
                while running:
                    try:
                        cam_sub.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass

                    rs, r_age = robot_buffer.interpolate_at(time.time_ns())
                    if rs is not None and r_age <= staleness_ms:
                        tqdm.write("Robot stream recovered. Resuming.")
                        break
                    time.sleep(0.05)
                continue

            # --- Build and publish synchronized observation ---
            # Single small JSON: cam_timestamp + robot_state + camera_meta.
            # The receiver pairs this with the camera image frame carrying
            # the same `timestamp_ns` (delivered directly from camera_node).
            t0 = time.perf_counter()
            broker_ns = time.time_ns()
            obs_meta = {
                "robot": {
                    "left": robot_state["left"],
                    "right": robot_state["right"],
                },
                "timestamps": {
                    "camera_ns": cam_timestamp_ns,
                    "robot_ns": robot_state["timestamp_ns"],
                    "broker_ns": broker_ns,
                },
                "camera_meta": {
                    "cameras": camera_names,
                    "height": cam_meta["height"],
                    "width": cam_meta["width"],
                    "channels": cam_meta["channels"],
                },
            }
            pub.send_json(obs_meta)
            t_send = time.perf_counter() - t0

            t_iter = time.perf_counter() - t_iter_start

            # Accumulate per-stage timings; print rolling avg every second.
            t_recv_us += t_recv * 1e6
            t_interp_us += t_interp * 1e6
            t_send_us += t_send * 1e6
            t_total_us += t_iter * 1e6
            timing_count += 1
            now = time.monotonic()
            if now - last_timing_report >= 1.0 and timing_count > 0:
                n = timing_count
                # "other" = total - measured stages = blocking-recv + bookkeeping
                other_us = t_total_us - (t_recv_us + t_interp_us + t_send_us)
                tqdm.write(
                    f"  [timing] {n} iters/s  avg per iter:  "
                    f"recv={t_recv_us/n:.0f}µs  "
                    f"interp={t_interp_us/n:.0f}µs  "
                    f"send={t_send_us/n:.0f}µs  "
                    f"other={other_us/n:.0f}µs  "
                    f"total={t_total_us/n:.0f}µs")
                t_recv_us = t_interp_us = t_send_us = t_total_us = 0.0
                timing_count = 0
                last_timing_report = now

            pbar.set_postfix_str(
                f"r_age={robot_age_ms:.1f}ms buf_r={len(robot_buffer)}")
            pbar.update(1)
            if args.debug:
                tqdm.write(
                    f"  robot interp@{robot_state['timestamp_ns']} "
                    f"cam@{cam_timestamp_ns}")

    except KeyboardInterrupt:
        pass
    finally:
        pbar.close()
        stop_event.set()
        robot_thread.join(timeout=1.0)
        cam_sub.close()
        robot_sub.close()
        pub.close()
        ctx.term()
        print("Sync broker shut down.")


if __name__ == "__main__":
    main()
