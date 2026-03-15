"""Camera node: publishes 3 USB webcam frames via ZMQ PUB at 60Hz.

Reads raw MJPG from USB cameras (skipping OpenCV's internal decode),
decodes in parallel threads, and publishes BGR frames over ZMQ.
Single decode in the entire pipeline — no re-encoding.

Usage:
    python -m yam_teleop.nodes.camera_node --config configs/camera.yaml
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm
import yaml
import zmq


class USBCamera:
    """Thin wrapper around OpenCV VideoCapture for a single USB camera."""

    def __init__(self, name: str, device_id: int, width: int, height: int,
                 fps: int, codec: str,
                 exposure: float, brightness: float,
                 saturation: float, contrast: float, sharpness: float,
                 white_balance: float):
        self.name = name
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{name}] Failed to open camera at device {device_id}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        self.cap.set(cv2.CAP_PROP_SATURATION, saturation)
        self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance)

        # Try raw MJPG passthrough (skip JPEG decode entirely)
        self._raw_jpeg = False
        if codec == "MJPG":
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            ok, raw = self.cap.read()
            if ok and raw is not None:
                flat = raw.flatten()
                if len(flat) >= 2 and flat[0] == 0xFF and flat[1] == 0xD8:
                    decoded = cv2.imdecode(flat, cv2.IMREAD_COLOR)
                    if decoded is not None and decoded.shape[:2] == (height, width):
                        self._raw_jpeg = True
                        print(f"[{name}] Initialized: device={device_id}, "
                              f"{width}x{height} @ {fps}Hz "
                              f"(raw JPEG passthrough, ~{len(flat)//1024}KB/frame)")
            if not self._raw_jpeg:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        if not self._raw_jpeg:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"[{name}] Camera opened but cannot read frames")
            print(f"[{name}] Initialized: device={device_id}, "
                  f"actual={frame.shape[1]}x{frame.shape[0]} @ {fps}Hz "
                  f"(OpenCV-decoded BGR; raw JPEG passthrough unavailable)")

    def read(self) -> np.ndarray:
        """Read one frame as a decoded BGR array.

        In raw JPEG mode, grabs the compressed buffer from V4L2 and
        decodes with imdecode (both steps release the GIL, so multiple
        cameras decode truly in parallel via ThreadPoolExecutor).
        """
        ok, data = self.cap.read()
        if not ok or data is None:
            raise RuntimeError(f"[{self.name}] Failed to read frame")
        if self._raw_jpeg:
            frame = cv2.imdecode(data.flatten(), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"[{self.name}] Failed to decode JPEG")
            return frame
        if (data.shape[1], data.shape[0]) != (self.width, self.height):
            data = cv2.resize(data, (self.width, self.height))
        return data

    def close(self):
        if self.cap is not None:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Camera node")
    parser.add_argument("--config", required=True, help="Path to camera.yaml")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable the live preview window")
    args = parser.parse_args()
    show_display = not args.no_display

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    width, height = cfg["resolution"]
    fps = cfg["fps"]
    codec = cfg.get("codec", "MJPG")
    port = cfg["zmq_port"]
    # Metadata-only port: same JSON header (timestamp + camera names) without
    # the image bytes. Used by the sync broker so it doesn't have to drain
    # ~36 MB/frame just to learn the camera timestamp.
    meta_port = cfg.get("meta_zmq_port", port + 4)

    # Initialize cameras
    cameras = OrderedDict()
    for name, cam_cfg in cfg["cameras"].items():
        cameras[name] = USBCamera(
            name=name,
            device_id=cam_cfg["device_id"],
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            exposure=cfg["exposure"],
            brightness=cfg["brightness"],
            saturation=cfg["saturation"],
            contrast=cfg["contrast"],
            sharpness=cfg["sharpness"],
            white_balance=cfg["white_balance"],
        )

    camera_names = list(cameras.keys())

    # ZMQ publishers — image stream and metadata-only stream.
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)
    pub.bind(f"tcp://127.0.0.1:{port}")
    meta_pub = ctx.socket(zmq.PUB)
    meta_pub.setsockopt(zmq.SNDHWM, 1)
    meta_pub.bind(f"tcp://127.0.0.1:{meta_port}")
    time.sleep(0.5)  # let subscribers connect

    # Graceful shutdown
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    executor = ThreadPoolExecutor(max_workers=len(cameras))

    # Display thread: shows BGR frames without blocking the main loop
    display_lock = threading.Lock()
    display_latest = {}  # camera_name -> BGR numpy array

    display_fps = [0.0]  # shared with main thread for tqdm

    def display_loop():
        display_w, display_h = width // 3, height // 3
        count = 0
        t0 = time.monotonic()
        while running:
            with display_lock:
                snapshot = dict(display_latest)
            if len(snapshot) < len(camera_names):
                time.sleep(0.01)
                continue
            resized = []
            for name in camera_names:
                small = cv2.resize(snapshot[name], (display_w, display_h))
                cv2.putText(small, name, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                resized.append(small)
            cv2.imshow("Camera Node", np.hstack(resized))
            cv2.waitKey(1)
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                display_fps[0] = count / (now - t0)
                count = 0
                t0 = now

    if show_display:
        display_thread = threading.Thread(target=display_loop, daemon=True)
        display_thread.start()

    print(f"Camera node publishing on :{port} (images) and :{meta_port} (meta) "
          f"at {fps}Hz{' (display ON)' if show_display else ''}")
    pbar = tqdm(desc="Camera node", unit="fr", smoothing=0.05, mininterval=1.0)

    try:
        while running:
            timestamp_ns = time.time_ns()

            # Read + decode all cameras in parallel
            futures = {name: executor.submit(cam.read)
                       for name, cam in cameras.items()}
            frames = {name: fut.result() for name, fut in futures.items()}

            # Publish multipart on image port: JSON metadata, then BGR bytes
            metadata = {
                "timestamp_ns": timestamp_ns,
                "cameras": camera_names,
                "height": height,
                "width": width,
                "channels": 3,
            }
            pub.send_json(metadata, flags=zmq.SNDMORE)
            for i, name in enumerate(camera_names):
                flag = zmq.SNDMORE if i < len(camera_names) - 1 else 0
                pub.send(frames[name].tobytes(), flags=flag)

            # Publish identical metadata on meta-only port (cheap, ~100 bytes).
            # Sync broker subscribes here so it doesn't have to drain image bytes.
            meta_pub.send_json(metadata)

            # Hand off decoded frames to display thread (non-blocking)
            if show_display:
                with display_lock:
                    display_latest.update(frames)
                pbar.set_postfix_str(f"display={display_fps[0]:.0f}fps")

            pbar.update(1)

    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        pbar.close()
        executor.shutdown(wait=False)
        for cam in cameras.values():
            cam.close()
        if show_display:
            cv2.destroyAllWindows()
        pub.close()
        meta_pub.close()
        ctx.term()
        print("Camera node shut down.")


if __name__ == "__main__":
    main()
