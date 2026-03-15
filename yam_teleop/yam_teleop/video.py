"""Streaming video writer for data collection.

Pipes BGR frames to ffmpeg subprocesses in real-time, one per camera.
No frames are held in memory — each is written to the ffmpeg stdin pipe
immediately and can be garbage collected.
"""

import subprocess
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
    `threads` caps ffmpeg's helper-thread count. 0 = auto (ffmpeg default).
    """
    codec: str = "libx264"
    container: str = "mp4"
    crf: int = 18
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    threads: int = 0

    def encoder_args(self) -> list[str]:
        """ffmpeg flags for this codec, including rate control."""
        args = ["-c:v", self.codec, "-preset", self.preset,
                "-pix_fmt", self.pix_fmt]
        if "nvenc" in self.codec:
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
    "h264":         VideoCodecConfig(crf=23, preset="medium"),
    "h264_hq":      VideoCodecConfig(crf=18, preset="medium"),
    "h264_fast":    VideoCodecConfig(crf=23, preset="fast"),
    # NVIDIA hardware encoder (NVENC). Encoding runs on the GPU; the small
    # remaining CPU cost is mostly BGR→NV12 color conversion. -threads 2
    # caps ffmpeg's helper threads so the broker / robot / gello / camera
    # nodes aren't starved at high camera counts.
    # NVENC presets: p1=fastest, p7=highest quality. p4 ~ libx264 medium.
    "nvenc":        VideoCodecConfig(codec="h264_nvenc", crf=28, preset="p4",
                                     threads=2),
    "nvenc_hq":     VideoCodecConfig(codec="h264_nvenc", crf=19, preset="p5",
                                     threads=2),
    "nvenc_fast":   VideoCodecConfig(codec="h264_nvenc", crf=23, preset="p2",
                                     threads=2),
}
DEFAULT_PRESET = "h264_hq"


class StreamingVideoWriter:
    """Streams BGR frames to per-camera ffmpeg subprocesses.

    Usage:
        writer = StreamingVideoWriter("/tmp/episode", codec=VIDEO_PRESETS["h264_hq"])
        writer.start(["top", "left_wrist"], width=1280, height=720)
        for obs in observations:
            for cam in obs["images"]:
                writer.write_frame(cam, obs["images"][cam])
        result = writer.finish()  # {cam: (path, frame_count)}
    """

    def __init__(self, output_dir: str | Path,
                 codec: VideoCodecConfig | None = None, fps: int = 60):
        self._output_dir = Path(output_dir)
        self._codec = codec or VideoCodecConfig()
        self._fps = fps
        self._procs: dict[str, subprocess.Popen] = {}
        self._paths: dict[str, Path] = {}
        self._stderr_paths: dict[str, Path] = {}
        self._stderr_files: dict[str, "Path"] = {}
        self._frame_counts: dict[str, int] = {}
        self._finished = False

    @property
    def is_started(self) -> bool:
        return len(self._procs) > 0

    def start(self, camera_names: list[str], width: int, height: int) -> None:
        """Open one ffmpeg subprocess per camera.

        ffmpeg's stdout/stderr go to log files (not subprocess.PIPE) — an
        unread PIPE fills its 64KB OS buffer in a few minutes of encoding,
        which blocks ffmpeg's stderr write and in turn hangs our stdin
        write_frame() call. Logging to files lets ffmpeg drain freely while
        keeping the diagnostic output for post-mortem on failure.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        c = self._codec
        for cam_name in camera_names:
            safe_name = cam_name.replace("/", "_")
            out_path = self._output_dir / f"{safe_name}.{c.container}"
            stderr_path = self._output_dir / f"{safe_name}.ffmpeg.log"
            cmd = [
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
            stderr_file = open(stderr_path, "wb")
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            self._procs[cam_name] = proc
            self._paths[cam_name] = out_path
            self._stderr_paths[cam_name] = stderr_path
            self._stderr_files[cam_name] = stderr_file
            self._frame_counts[cam_name] = 0

    def write_frame(self, cam_name: str, frame: np.ndarray) -> None:
        """Write one BGR frame to the camera's ffmpeg process."""
        self._procs[cam_name].stdin.write(frame.tobytes())
        self._frame_counts[cam_name] += 1

    def finish(self) -> dict[str, tuple[Path, int]]:
        """Close all ffmpeg processes and return {cam: (path, frame_count)}.

        Raises RuntimeError if any ffmpeg process exits with an error.
        """
        if self._finished:
            return {cam: (self._paths[cam], self._frame_counts[cam])
                    for cam in self._procs}

        errors = []
        for cam_name, proc in self._procs.items():
            proc.stdin.close()
        for cam_name, proc in self._procs.items():
            proc.wait(timeout=120)
            self._stderr_files[cam_name].close()
            if proc.returncode != 0:
                try:
                    stderr = self._stderr_paths[cam_name].read_bytes().decode(
                        errors="replace")
                except Exception:
                    stderr = "(could not read stderr log)"
                errors.append(f"{cam_name}: ffmpeg exit {proc.returncode}\n{stderr}")

        # Successful runs: drop the per-camera log file to keep episode dirs clean
        if not errors:
            for stderr_path in self._stderr_paths.values():
                try:
                    stderr_path.unlink()
                except Exception:
                    pass

        self._finished = True
        if errors:
            raise RuntimeError("Video encoding failed:\n" + "\n".join(errors))

        return {cam: (self._paths[cam], self._frame_counts[cam])
                for cam in self._procs}

    def discard(self) -> None:
        """Kill all ffmpeg processes and delete video files."""
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        for f in self._stderr_files.values():
            try:
                f.close()
            except Exception:
                pass
        for path in self._paths.values():
            if path.exists():
                path.unlink()
        for path in self._stderr_paths.values():
            if path.exists():
                path.unlink()
        self._procs.clear()
        self._paths.clear()
        self._stderr_paths.clear()
        self._stderr_files.clear()
        self._frame_counts.clear()
        self._finished = False

    def cleanup(self) -> None:
        """Defensive cleanup for finally blocks. Never raises."""
        try:
            self.discard()
        except Exception:
            pass
