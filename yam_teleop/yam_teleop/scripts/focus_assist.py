"""Live focus assist: prints a sharpness score for one camera stream.

Twist the camera's lens barrel until the number (and bar) peaks.
Reads the camera node's ZMQ stream, so the node must be running and the
camera cannot be opened by anything else.

Usage:
    python -m yam_teleop.scripts.focus_assist --camera left_wrist
"""

import argparse
import time

import cv2
import numpy as np
import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="left_wrist",
                        help="Camera name as published by the camera node")
    parser.add_argument("--port", type=int, default=5001,
                        help="Camera node ZMQ pub port")
    args = parser.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    sub.connect(f"tcp://127.0.0.1:{args.port}")

    peak = 0.0
    print(f"Focus assist for '{args.camera}' — twist the lens until the bar peaks. Ctrl-C to quit.")
    while True:
        try:
            meta = sub.recv_json()
            parts = [sub.recv() for _ in range(len(meta["cameras"]))]
        except zmq.Again:
            print("  (no frames — is the camera node running?)")
            continue
        if args.camera not in meta["cameras"]:
            raise SystemExit(f"'{args.camera}' not in stream: {meta['cameras']}")
        i = meta["cameras"].index(args.camera)
        img = np.frombuffer(parts[i], np.uint8).reshape(
            meta["height"], meta["width"], meta["channels"])
        h, w = img.shape[:2]
        crop = cv2.cvtColor(img[h // 4:3 * h // 4, w // 4:3 * w // 4],
                            cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(crop, cv2.CV_64F).var()
        peak = max(peak, score)
        bar = "#" * min(60, int(score))
        print(f"\r  sharpness={score:7.1f}  peak={peak:7.1f}  |{bar:<60s}|",
              end="", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
