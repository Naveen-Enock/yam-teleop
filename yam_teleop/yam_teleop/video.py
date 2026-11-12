"""Streaming video writer for data collection.

One ffmpeg subprocess per camera, each fed by its own writer thread through a
bounded queue. The teleop loop never waits on the encoder:

  * write_frame() only enqueues a reference to the frame (no copy for the
    read-only buffers env._get_obs() hands out) and returns immediately.
    The writer thread pushes the bytes into ffmpeg's stdin pipe; the pipe
    write releases the GIL, so the control loop keeps running.
  * Encoding defaults to NVIDIA NVENC (GPU). Frames go in as bgr24, ffmpeg
    repacks them to bgr0 (a cheap byte shuffle) and NVENC does the
    RGB→YUV 4:2:0 conversion and H.264 encode on the GPU. The CPU presets
    (libx264) are kept for machines without an NVIDIA GPU, but at
    3 × 1080p60 they saturate the CPU and starve the robot / leader / broker
    nodes — that contention, plus the old synchronous pipe write in the
    control loop, is what showed up as teleop jitter.
  * ffmpeg runs under `nice` so whatever CPU work it does have (demuxing the
    pipe, the bgr24→bgr0 repack) never preempts the control nodes.

If the encoder falls behind, frames buffer in the queue (default: 1 s per
camera). Only if that buffer fills does write_frame() block — frames are
never dropped, so the video stays frame-aligned with the HDF5 step index —
and a warning is printed. peak_backlog() exposes the queue depth for the
collect scripts' [diag] line.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoCodecConfig:
    """Configuration for ffmpeg video encoding.

    `crf` is the libx264 constant rate factor (0–51, lower=better).
    For NVENC it's reused as the `-cq` constant-quality target.
    `preset` strings differ between encoders:
      - libx264: ultrafast/superfast/veryfast/faster/fast/medium/slow/...
      - h264_nvenc: p1 (fastest) ... p7 (slowest/highest quality)
    `pix_fmt` is the format handed to the encoder. For NVENC use "bgr0" so
      the RGB→YUV conversion happens on the GPU (the bitstream is still
      standard yuv420p H.264); "yuv420p" would make ffmpeg do that
      conversion on the CPU first.
    `threads` caps ffmpeg's helper-thread count. 0 = auto (ffmpeg default).
    `nice` is the niceness ffmpeg runs at (0 = same priority as us).
    """
    codec: str = "libx264"
    container: str = "mp4"
    crf: int = 18
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    threads: int = 0
    nice: int = 10

    @property
    def is_gpu(self) -> bool:
        return "nvenc" in self.codec

    def encoder_args(self) -> list[str]:
        """ffmpeg flags for this codec, including rate control."""
        args = ["-c:v", self.codec, "-preset", self.preset,
                "-pix_fmt", self.pix_fmt]
        if self.is_gpu:
            # GPU encode: VBR with constant-quality target. -b:v 0 means
            # no bitrate cap; quality is governed entirely by -cq.
            args += ["-rc", "vbr", "-cq", str(self.crf), "-b:v", "0"]
        else:
            # libx264 / libx265 / etc.
            args += ["-crf", str(self.crf)]
        if self.threads > 0:
            args += ["-threads", str(self.threads)]
        return args


VIDEO_PRESETS = {
    # NVIDIA hardware encoder (NVENC) — the default. Encode and colour
    # conversion both run on the GPU (pix_fmt bgr0, see VideoCodecConfig).
    # NVENC presets: p1=fastest, p7=highest quality. p4 ~ libx264 medium.
    # Each camera is one NVENC session; 3 cameras fits every GeForce
    # driver's concurrent-session limit.
    "nvenc_hq":     VideoCodecConfig(codec="h264_nvenc", crf=19, preset="p5",
                                     pix_fmt="bgr0", threads=2),
    "nvenc":        VideoCodecConfig(codec="h264_nvenc", crf=28, preset="p4",
                                     pix_fmt="bgr0", threads=2),
    "nvenc_fast":   VideoCodecConfig(codec="h264_nvenc", crf=23, preset="p2",
                                     pix_fmt="bgr0", threads=2),
    # CPU (libx264) fallbacks for machines without an NVIDIA GPU. At
    # 3 × 1080p60 these compete with the control nodes for CPU and cause
    # teleop jitter — use only for dev / low camera load.
    "h264":         VideoCodecConfig(crf=23, preset="medium"),
    "h264_hq":      VideoCodecConfig(crf=18, preset="medium"),
    "h264_fast":    VideoCodecConfig(crf=23, preset="fast"),
}
DEFAULT_PRESET = "nvenc_hq"


def check_encoder(codec: VideoCodecConfig, timeout: float = 30.0) -> None:
    """Run a tiny test encode; raise RuntimeError if this codec can't work here.

    Call once at startup so a missing GPU / driver / NVENC-less ffmpeg build
    fails before the first episode instead of mid-recording.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (sudo apt install ffmpeg)")
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=640x480:r=60",
        "-frames:v", "10",
        *codec.encoder_args(),
        "-f", "null", "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg {codec.codec} test encode timed out "
                           f"after {timeout:.0f}s")
    if res.returncode != 0:
        err = res.stderr.decode(errors="replace").strip()
        hint = ""
        if codec.is_gpu:
            hint = ("\n  NVENC needs an NVIDIA GPU, the proprietary driver "
                    "(check `nvidia-smi`), and an ffmpeg built with nvenc "
                    "(`ffmpeg -encoders | grep nvenc`). Close other programs "
                    "holding NVENC sessions. As a last resort pass "
                    "--video-codec h264_fast (CPU; expect teleop jitter).")
        raise RuntimeError(f"ffmpeg {codec.codec} test encode failed "
                           f"(exit {res.returncode}):\n{err}{hint}")


_STOP = object()


class _CameraStream:
    """One camera: ffmpeg process + writer thread + bounded frame queue."""

    def __init__(self, name: str, proc: subprocess.Popen, path: Path,
                 log_path: Path, log_file, max_queue: int):
        self.name = name
        self.proc = proc
        self.path = path
        self.log_path = log_path
        self.log_file = log_file
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.error: BaseException | None = None
        self.aborted = False
        self.frames_enqueued = 0
        self.frames_written = 0
        self.peak_backlog = 0
        self.overflows = 0
        self.thread = threading.Thread(
            target=self._run, name=f"video-{name}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        stdin = self.proc.stdin
        while True:
            item = self.queue.get()
            if item is _STOP:
                break
            if self.error is not None or self.aborted:
                continue  # keep draining so producers never block forever
            try:
                stdin.write(item)  # releases the GIL while the pipe drains
                self.frames_written += 1
            except BaseException as e:  # BrokenPipeError if ffmpeg died
                self.error = e
        try:
            stdin.close()  # EOF -> ffmpeg finalizes the file
        except BaseException as e:
            if self.error is None and not self.aborted:
                self.error = e


class StreamingVideoWriter:
    """Streams BGR frames to per-camera ffmpeg subprocesses, off-thread.

    Usage:
        writer = StreamingVideoWriter("/tmp/episode", codec=VIDEO_PRESETS["nvenc_hq"])
        writer.start(["top", "left_wrist"], width=1280, height=720)
        for obs in observations:
            for cam in obs["images"]:
                writer.write_frame(cam, obs["images"][cam])
        result = writer.finish()  # {cam: (path, frame_count)}

    `max_queue_s` is how many seconds of frames each camera may buffer while
    the encoder catches up before write_frame() starts blocking.
    """

    def __init__(self, output_dir: str | Path,
                 codec: VideoCodecConfig | None = None, fps: int = 60,
                 max_queue_s: float = 1.0):
        self._output_dir = Path(output_dir)
        self._codec = codec or VideoCodecConfig()
        self._fps = fps
        self._max_queue = max(1, int(round(max_queue_s * fps)))
        self._streams: dict[str, _CameraStream] = {}
        self._frame_nbytes = 0
        self._finished = False

    @property
    def is_started(self) -> bool:
        return len(self._streams) > 0

    def start(self, camera_names: list[str], width: int, height: int) -> None:
        """Open one ffmpeg subprocess (and writer thread) per camera.

        ffmpeg's stdout/stderr go to log files (not subprocess.PIPE) — an
        unread PIPE fills its 64KB OS buffer in a few minutes of encoding,
        which blocks ffmpeg's stderr write and in turn hangs our stdin
        write. Logging to files lets ffmpeg drain freely while keeping the
        diagnostic output for post-mortem on failure.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_nbytes = width * height * 3
        c = self._codec
        nice = []
        if c.nice > 0 and shutil.which("nice"):
            nice = ["nice", "-n", str(c.nice)]
        for cam_name in camera_names:
            safe_name = cam_name.replace("/", "_")
            out_path = self._output_dir / f"{safe_name}.{c.container}"
            log_path = self._output_dir / f"{safe_name}.ffmpeg.log"
            cmd = [
                *nice,
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(self._fps),
                "-i", "pipe:0",
                *c.encoder_args(),
                str(out_path),
            ]
            log_file = open(log_path, "wb")
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
            )
            self._streams[cam_name] = _CameraStream(
                cam_name, proc, out_path, log_path, log_file, self._max_queue)

    def write_frame(self, cam_name: str, frame: np.ndarray) -> None:
        """Queue one BGR frame for the camera's encoder. Normally non-blocking.

        Read-only frames (what env._get_obs() returns — views of immutable
        ZMQ bytes) are queued by reference; writable arrays are copied first
        so the caller may reuse them.
        """
        s = self._streams[cam_name]
        if s.error is not None:
            raise RuntimeError(
                f"video writer for {cam_name!r} failed: {s.error!r}\n"
                f"{self._read_log(s)}")
        if frame.nbytes != self._frame_nbytes or frame.dtype != np.uint8:
            raise ValueError(
                f"{cam_name}: frame {frame.shape} {frame.dtype} doesn't match "
                f"the {self._frame_nbytes}-byte uint8 BGR frames this writer "
                f"was started with")
        if frame.flags.writeable or not frame.flags.c_contiguous:
            frame = np.array(frame, order="C", copy=True)
        buf = memoryview(frame).cast("B")

        try:
            s.queue.put_nowait(buf)
        except queue.Full:
            # Encoder can't keep up. Block rather than drop so video frames
            # stay aligned 1:1 with recorded steps.
            s.overflows += 1
            if s.overflows == 1:
                print(f"  [video] WARNING: {cam_name} encoder backlog full "
                      f"({self._max_queue} frames) — recording loop is now "
                      f"throttled by video encoding. Use a faster "
                      f"--video-codec preset.", file=sys.stderr)
            s.queue.put(buf)
        s.frames_enqueued += 1
        depth = s.queue.qsize()
        if depth > s.peak_backlog:
            s.peak_backlog = depth

    def peak_backlog(self, reset: bool = True) -> int:
        """Largest per-camera queue depth (frames) since the last reset."""
        peak = max((s.peak_backlog for s in self._streams.values()), default=0)
        if reset:
            for s in self._streams.values():
                s.peak_backlog = s.queue.qsize()
        return peak

    @property
    def overflows(self) -> int:
        """Times write_frame() had to block because a queue was full."""
        return sum(s.overflows for s in self._streams.values())

    def finish(self) -> dict[str, tuple[Path, int]]:
        """Flush queued frames, close ffmpeg, return {cam: (path, frame_count)}.

        Raises RuntimeError if any camera's writer or ffmpeg process failed.
        """
        if self._finished:
            return self._result()

        pending = sum(s.queue.qsize() for s in self._streams.values())
        if pending > self._fps:
            print(f">> Flushing {pending} buffered video frames...")
        for s in self._streams.values():
            s.queue.put(_STOP)

        errors = []
        join_timeout = 120.0 + pending / max(self._fps, 1)
        for s in self._streams.values():
            s.thread.join(timeout=join_timeout)
            if s.thread.is_alive():
                # Writer wedged on a hung ffmpeg; kill it to unblock the write.
                s.proc.kill()
                s.thread.join(timeout=5)
                errors.append(f"{s.name}: writer timed out after "
                              f"{join_timeout:.0f}s")
        for s in self._streams.values():
            try:
                s.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                s.proc.kill()
                s.proc.wait(timeout=5)
            s.log_file.close()
            if s.error is not None:
                errors.append(f"{s.name}: write failed: {s.error!r}\n"
                              f"{self._read_log(s)}")
            elif s.proc.returncode != 0:
                errors.append(f"{s.name}: ffmpeg exit {s.proc.returncode}\n"
                              f"{self._read_log(s)}")

        # Successful runs: drop the per-camera log file to keep episode dirs clean
        if not errors:
            for s in self._streams.values():
                try:
                    s.log_path.unlink()
                except Exception:
                    pass

        self._finished = True
        if errors:
            raise RuntimeError("Video encoding failed:\n" + "\n".join(errors))
        return self._result()

    def discard(self) -> None:
        """Kill all ffmpeg processes and delete video files."""
        for s in self._streams.values():
            s.aborted = True
            if s.proc.poll() is None:
                s.proc.kill()
        for s in self._streams.values():
            try:
                s.proc.wait(timeout=5)
            except Exception:
                pass
            # The thread skips writes once aborted, so the queue drains fast.
            try:
                s.queue.put(_STOP, timeout=5)
            except queue.Full:
                pass
            s.thread.join(timeout=5)
            try:
                s.log_file.close()
            except Exception:
                pass
            for path in (s.path, s.log_path):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        self._streams.clear()
        self._finished = False

    def cleanup(self) -> None:
        """Defensive cleanup for finally blocks. Never raises."""
        try:
            self.discard()
        except Exception:
            pass

    def _result(self) -> dict[str, tuple[Path, int]]:
        return {name: (s.path, s.frames_written)
                for name, s in self._streams.items()}

    @staticmethod
    def _read_log(s: _CameraStream) -> str:
        try:
            s.log_file.flush()
        except Exception:
            pass
        try:
            return s.log_path.read_bytes().decode(errors="replace")
        except Exception:
            return "(could not read ffmpeg log)"
