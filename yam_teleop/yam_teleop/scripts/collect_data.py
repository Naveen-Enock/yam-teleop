"""Data collection script using GELLO teleoperation.

Cycle:
1. Reset YAM arms (open grippers, then close)
2. Reset GELLO arms to neutral (hold until both grippers squeezed)
3. Teleop + recording starts
4. Left foot pedal = save as FAILURE, Right foot pedal = save as SUCCESS
   'q' to quit

Teleop is ONLY active while recording. Between episodes the arms hold position.
Video frames are streamed to ffmpeg in real-time (no in-memory accumulation).

Usage:
    python -m yam_teleop.scripts.collect_data --env-config configs/env.yaml --output-dir data/task_name
"""

import argparse
import atexit
import os
import select
import shutil
import sys
import tempfile
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import zmq

from yam_teleop.env import YAMBimanualEnv
from yam_teleop.foot_pedal import FootPedalHub, KEY_AUDIO, KEY_FAILURE, KEY_SUCCESS
from yam_teleop.scripts.record_audio import AudioRecorder
from yam_teleop.video import (
    DEFAULT_PRESET,
    VIDEO_PRESETS,
    StreamingVideoWriter,
)

# GELLO gripper value must be below this to count as "closed" (trigger squeezed)
GRIPPER_CLOSE_THRESHOLD = 0.1


def plot_debug_episode(debug_data: dict, episode_dir: str) -> None:
    """Plot GELLO vs YAM joint positions for one episode and save to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = 60
    steps = len(debug_data["gello_right"])
    t = np.arange(steps) / fps

    gello_r = np.array(debug_data["gello_right"])
    yam_r = np.array(debug_data["yam_right"])
    gello_l = np.array(debug_data["gello_left"])
    yam_l = np.array(debug_data["yam_left"])

    for side, gello, yam in [("right", gello_r, yam_r),
                              ("left", gello_l, yam_l)]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"{side.upper()} arm — GELLO vs YAM joint positions",
                     fontsize=14)

        for j in range(6):
            ax = axes[j // 2, j % 2]
            ax.plot(t, np.degrees(yam[:, j]), label="YAM", linewidth=1)
            ax.plot(t, np.degrees(gello[:, j]), label="GELLO",
                    linewidth=1, linestyle="--")
            diff = gello[:, j] - yam[:, j]
            ax.fill_between(t, np.degrees(yam[:, j]),
                            np.degrees(gello[:, j]),
                            alpha=0.15, color="red")

            max_diff_deg = np.max(np.abs(np.degrees(diff)))
            ax.set_ylabel("degrees")
            ax.set_title(f"Joint {j}  (max |diff| = {max_diff_deg:.1f} deg)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("time (s)")
        axes[-1, 1].set_xlabel("time (s)")
        fig.tight_layout()

        path = os.path.join(episode_dir, f"debug_{side}_joints.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f">> Debug plot saved: {path}")


class KeyListener:
    """Async single-keystroke listener using raw terminal mode."""

    def __init__(self):
        self._key = None
        self._lock = threading.Lock()
        self._original_settings = termios.tcgetattr(sys.stdin)
        self._closed = False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        # Restore termios on any exit path (SystemExit before the main
        # try/finally, uncaught exceptions, etc.) — otherwise stdin stays
        # in cbreak mode and the user's shell loses echo after we die.
        atexit.register(self.close)

    def _read_loop(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._key = ch.lower()
        except Exception:
            pass

    def get_key(self) -> str | None:
        with self._lock:
            k = self._key
            self._key = None
            return k

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._running = False
        self._thread.join(timeout=1.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._original_settings)


def both_gello_grippers_closed(gello_msg: dict, threshold: float) -> bool:
    """True when both GELLO grippers are squeezed past the threshold.

    Takes a raw gello_node message (from env.get_latest_gello()).
    """
    left = float(gello_msg["left"]["gripper_pos"])
    right = float(gello_msg["right"]["gripper_pos"])
    return left < threshold and right < threshold


def gello_msg_to_action(gello_msg: dict) -> tuple[np.ndarray, dict, int]:
    """Convert a gello_node state message to (action_vector, components, ts).

    The action vector is the concatenation of left joints, left gripper,
    right joints, right gripper — same order written to HDF5. components
    holds each piece keyed by the same names as ``episode["gello"]`` so
    callers don't have to re-slice the action vector to record per-arm
    fields. Hoisted into one place so the action layout can change without
    touching multiple call sites.
    """
    left_jp = np.asarray(gello_msg["left"]["joint_pos"])
    left_gp = np.asarray([float(gello_msg["left"]["gripper_pos"])])
    right_jp = np.asarray(gello_msg["right"]["joint_pos"])
    right_gp = np.asarray([float(gello_msg["right"]["gripper_pos"])])
    action = np.concatenate([left_jp, left_gp, right_jp, right_gp])
    components = {
        "left/joint_pos": left_jp,
        "left/gripper_pos": left_gp,
        "right/joint_pos": right_jp,
        "right/gripper_pos": right_gp,
    }
    return action, components, int(gello_msg["timestamp_ns"])


def reset_gello(env, gello_cmd, home_joints_left, home_joints_right,
                gripper_threshold=0.1, joint_speed=0.2, send_hz=200.0,
                keys=None):
    """Reset GELLO arms to home position.

    Streams waypoints to gello_cmd at send_hz (decoupled from the 60Hz
    broker) so motion is smooth and fast. joint_speed (rad/s) caps the
    largest joint's slew rate; smaller joints arrive proportionally faster.

    All commands go through one PUSH socket (gello_cmd).
    """
    # Enable torque
    gello_cmd.send_json({"command": "enable_torque", "gripper_limp": True})

    # Wait for a fresh gello message after torque takes effect
    gello_msg = None
    deadline = time.monotonic() + 2.0
    while gello_msg is None and time.monotonic() < deadline:
        gello_msg = env.get_latest_gello()
        time.sleep(0.005)
    if gello_msg is None:
        raise RuntimeError("No gello messages received after enable_torque")

    current_left = np.asarray(gello_msg["left"]["joint_pos"])
    current_right = np.asarray(gello_msg["right"]["joint_pos"])
    target_left = np.array(home_joints_left)
    target_right = np.array(home_joints_right)

    # Interpolate to home; cap slew rate at joint_speed rad/s.
    max_d = max(np.max(np.abs(current_left - target_left)),
                np.max(np.abs(current_right - target_right)))
    duration = max(max_d / joint_speed, 0.05)
    num_steps = max(int(duration * send_hz), 1)
    print(f"  GELLO reset: {num_steps} steps over {duration:.2f}s "
          f"(max_delta={max_d:.3f} rad)")
    period = 1.0 / send_hz
    next_t = time.monotonic()
    for lw, rw in zip(
        np.linspace(current_left, target_left, num_steps + 1)[1:],
        np.linspace(current_right, target_right, num_steps + 1)[1:],
    ):
        gello_cmd.send_json({
            "left": {"joint_pos": lw.tolist(), "gripper_pos": 0.0},
            "right": {"joint_pos": rw.tolist(), "gripper_pos": 0.0},
        })
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.monotonic()

    # Hold and wait for grippers (like env._hold_until_converged).
    # Polls the live gello stream at ~200Hz so gripper-close detection is
    # responsive — this latency directly contributes to the lag between
    # the operator squeezing and YAM beginning to follow.
    print("  GELLO at neutral. Squeeze both grippers to start...")
    while True:
        gello_cmd.send_json({
            "left": {"joint_pos": target_left.tolist(), "gripper_pos": 0.0},
            "right": {"joint_pos": target_right.tolist(), "gripper_pos": 0.0},
        })
        gello_msg = env.get_latest_gello()
        if gello_msg is not None and both_gello_grippers_closed(
                gello_msg, gripper_threshold):
            break
        time.sleep(0.005)
        if keys is not None:
            key = keys.get_key()
            if key == "q":
                print(">> Quitting (q in reset_gello)")
                raise SystemExit("reset_gello-q")

    # Disable torque
    gello_cmd.send_json({"command": "disable_torque"})


def save_episode(path: str, episode: dict, video_info: dict,
                 codec_config, success: bool = False) -> None:
    """Save one episode to HDF5 with video file references."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "actions", data=np.array(episode["actions"]), compression="gzip",
        )

        ts = f.create_group("timestamps")
        ts.create_dataset("camera_ns", data=np.array(episode["camera_ns"]))
        ts.create_dataset("robot_ns", data=np.array(episode["robot_ns"]))
        ts.create_dataset("gello_ns", data=np.array(episode["gello_ns"]))
        ts.create_dataset("broker_ns", data=np.array(episode["broker_ns"]))

        robot = f.create_group("robot")
        for key in ["left/joint_pos", "left/joint_vel", "left/joint_eff",
                     "left/gripper_pos", "left/gripper_eff",
                     "right/joint_pos", "right/joint_vel", "right/joint_eff",
                     "right/gripper_pos", "right/gripper_eff"]:
            robot.create_dataset(
                key, data=np.array(episode["robot"][key]), compression="gzip",
            )

        gello = f.create_group("gello")
        for key in ["left/joint_pos", "left/gripper_pos",
                     "right/joint_pos", "right/gripper_pos"]:
            gello.create_dataset(
                key, data=np.array(episode["gello"][key]), compression="gzip",
            )

        images = f.create_group("images")
        for cam_name, (video_filename, num_frames) in video_info.items():
            cam_group = images.create_group(cam_name)
            cam_group.attrs["video_file"] = video_filename
            cam_group.attrs["codec"] = codec_config.codec
            cam_group.attrs["num_frames"] = num_frames
            cam_group.attrs["fps"] = 60

        f.attrs["num_steps"] = len(episode["actions"])
        f.attrs["success"] = success
        f.attrs["image_storage"] = "video"
        f.attrs["created_at"] = datetime.now().isoformat()
        # Vestigial: the converter derives obs/action alignment from the logged
        # timestamps (broker_ns vs gello_ns), not this attr (fallback only for
        # timestamp-less files). See OBS_ACTION_ALIGNMENT.md.
        f.attrs["obs_action_aligned"] = True


def main():
    parser = argparse.ArgumentParser(description="Collect teleop data")
    parser.add_argument("--env-config", required=True, help="Path to env.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to save episodes")
    parser.add_argument("--max-steps", type=int, default=18000,
                        help="Max steps per episode (default: 18000 = 5min at 60Hz)")
    parser.add_argument("--gripper-threshold", type=float, default=GRIPPER_CLOSE_THRESHOLD,
                        help="GELLO gripper close threshold (default: 0.1)")
    parser.add_argument("--gello-cmd-port", type=int, default=5006,
                        help="ZMQ port for GELLO commands (default: 5006)")
    parser.add_argument("--video-codec", default=DEFAULT_PRESET,
                        choices=list(VIDEO_PRESETS.keys()),
                        help=f"Video codec preset (default: {DEFAULT_PRESET})")
    parser.add_argument("--debug-plot", action="store_true",
                        help="Plot GELLO vs YAM joint positions after each episode")
    parser.add_argument("--skip-mic-check", action="store_true",
                        help="Skip the startup microphone alive-check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    task_name = Path(args.output_dir).name
    codec_config = VIDEO_PRESETS[args.video_codec]

    env = YAMBimanualEnv(args.env_config)

    # GELLO command socket (PUSH -> gello_node PULL)
    gello_cmd_ctx = zmq.Context()
    gello_cmd = gello_cmd_ctx.socket(zmq.PUSH)
    gello_cmd.setsockopt(zmq.LINGER, 0)
    gello_cmd.connect(f"tcp://127.0.0.1:{args.gello_cmd_port}")
    time.sleep(0.5)

    # Foot pedals (grabs all FootSwitch devices)
    hub = FootPedalHub()

    # Audio recorder (background thread with audio pedal)
    audio_state = {"step": -1, "episode": 0, "episode_dir": None}
    audio_recorder = AudioRecorder(audio_state, pedal=hub.channel(KEY_AUDIO))
    audio_recorder.start()

    keys = KeyListener()
    episode_count = 0
    video_writer = None
    tmp_dir = None

    print("\nData Collection Cycle:")
    print("  1. YAM arms reset (grippers open then close)")
    print("  2. GELLO arms reset to neutral (hold until grippers squeezed)")
    print("  3. Teleop + recording starts (audio records continuously)")
    print("  4. Audio pedal = mark sub-task transition (timestamped marker)")
    print("  5. Left pedal = save as FAILURE, Right pedal = save as SUCCESS")
    print("     'q' to quit")
    print(f"  Gripper close threshold: {args.gripper_threshold}")
    print(f"  GELLO cmd port: {args.gello_cmd_port}")
    print(f"  Video codec: {args.video_codec} "
          f"({codec_config.codec} crf={codec_config.crf} "
          f"preset={codec_config.preset})\n")

    # Microphone alive-check before the first episode. The recorder's
    # rolling-window peak detector needs ~5s of samples to render a verdict;
    # block here so we don't take the operator through a YAM/GELLO reset
    # only to refuse to start recording.
    if args.skip_mic_check:
        print("Skipping microphone check (--skip-mic-check)\n")
    else:
        print("Verifying microphone...")
        if not audio_recorder.wait_for_audio_check():
            print("ERROR: microphone is silent / unresponsive at startup. "
                  "Check battery / mute / USB connection and relaunch.")
            raise SystemExit("audio-startup-check-failed")
        print(">> Microphone OK\n")

    try:
        while True:
            # Block before any episode setup if the mic ever went silent
            # this session. Sticky — only a script restart clears it.
            if audio_recorder.is_audio_dead():
                print("ERROR: microphone went silent during a previous "
                      "episode. Audio for that episode is incomplete and "
                      "future episodes are BLOCKED. Please fix the mic "
                      "and relaunch the data collection script.")
                raise SystemExit("audio-died-mid-session")

            # === PHASE 1: RESET YAM ===
            obs, info = env.reset()

            # === PHASE 2: RESET GELLO ARMS ===
            print("Resetting GELLO (move to neutral, "
                  "squeeze both grippers to start)...")
            reset_gello(
                env, gello_cmd,
                home_joints_left=env._home_position[:6].tolist(),
                home_joints_right=env._home_position[7:13].tolist(),
                gripper_threshold=args.gripper_threshold,
                keys=keys,
            )
            print(">> GELLO ready")

            # === PHASE 3: TELEOP + RECORD ===
            # Wait for the first gello message — it's the action source for
            # every step. CONFLATE on env._gello_sub means we always get the
            # freshest sample; the wait here is just for first arrival.
            t0 = time.time()
            gello_msg = None
            deadline = time.monotonic() + 5.0
            while gello_msg is None and time.monotonic() < deadline:
                gello_msg = env.get_latest_gello()
                if gello_msg is None:
                    time.sleep(0.005)
            if gello_msg is None:
                raise RuntimeError("No gello messages within 5s of teleop start")
            print(f">> First gello received ({time.time()-t0:.3f}s)")

            # Drain to a fresh broker obs so the first recorded step reflects
            # post-reset_gello state, not anything emitted while we were
            # blocking on operator gripper-squeeze. CONFLATE on _obs_sub
            # makes this a single non-blocking read; the timeout just guards
            # against a halted broker.
            t0 = time.time()
            warmup_obs = env.poll_broker_obs(timeout_ms=2000)
            if warmup_obs is None:
                raise RuntimeError("No broker obs within 2s of teleop start")
            print(f">> First synchronized obs received ({time.time()-t0:.3f}s)")
            print(">> Teleop + recording started")

            # Create episode directory
            episode_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"{task_name}_{episode_timestamp}"
            episode_dir = os.path.join(args.output_dir, episode_name)
            os.makedirs(episode_dir, exist_ok=True)

            audio_state["step"] = 0
            audio_state["episode"] = episode_count
            audio_state["episode_dir"] = episode_dir

            # Temp directory for video files during recording
            tmp_dir = tempfile.mkdtemp(dir=episode_dir,
                                       prefix=".tmp_video_")
            video_writer = StreamingVideoWriter(
                tmp_dir, codec=codec_config, fps=60)

            # Recorded gello fields. The action vector is the concatenation
            # of these in the same order — see gello_msg_to_action().
            gello_record_keys = [
                "left/joint_pos", "left/gripper_pos",
                "right/joint_pos", "right/gripper_pos",
            ]
            episode = {
                "actions": [],
                "camera_ns": [], "robot_ns": [],
                "gello_ns": [], "broker_ns": [],
                "robot": {k: [] for k in warmup_obs["robot"]},
                "gello": {k: [] for k in gello_record_keys},
            }

            debug_data = {
                "gello_right": [], "yam_right": [],
                "gello_left": [], "yam_left": [],
            } if args.debug_plot else None

            hub.clear_pending()  # discard stale pedal presses from reset
            episode_outcome = None  # True=success, False=failure

            # Single-rate teleop loop:
            #   1. Read the freshest gello message (CONFLATE keeps latest).
            #   2. env.step(action) sends the command to robot_node and
            #      blocks for the next synchronized broker obs (~16.7ms).
            #      robot_node CONFLATEs on its receive side and re-applies
            #      the latest target to YAM at ~250Hz internally, so 60Hz
            #      publish from here is plenty.
            #   3. Record one episode frame.
            # Same env.step() is used by run_policy.py / replay_episode.py,
            # so the teleop and policy paths share the gym contract.
            t0 = time.time()
            last_diag_t = time.monotonic()
            recorded_at_last_diag = 0
            # Record the obs in hand + this step's action + the four timestamps; the
            # converter derives the obs/action alignment from those timestamps (this
            # pre-step convention resolves to shift 0, the post-command one to shift 1
            # -- both land on the same conditioning-obs pairing). `obs` starts as the
            # post-reset warmup obs and is advanced by env.step at the end of each
            # iteration. See OBS_ACTION_ALIGNMENT.md.
            obs = warmup_obs
            while True:
                # gello publishes at 200Hz vs our ~60Hz step, so a fresh
                # sample is virtually always waiting. On the rare iteration
                # where it isn't (gello hiccup), we re-use the previous
                # message rather than skip the step — keeps loop rate steady.
                fresh = env.get_latest_gello()
                if fresh is not None:
                    gello_msg = fresh

                action, components, gello_ns = gello_msg_to_action(gello_msg)

                # Record one episode frame from the obs in hand + this action + its
                # timestamps. The obs/action alignment is set later by the converter
                # from those timestamps, not by which obs is recorded here.
                episode["actions"].append(action.copy())
                episode["camera_ns"].append(obs["timestamps"]["camera_ns"])
                episode["robot_ns"].append(obs["timestamps"]["robot_ns"])
                episode["gello_ns"].append(gello_ns)
                episode["broker_ns"].append(obs["timestamps"]["broker_ns"])
                for k in episode["robot"]:
                    episode["robot"][k].append(obs["robot"][k].copy())
                for k in gello_record_keys:
                    episode["gello"][k].append(components[k].copy())

                if debug_data is not None:
                    debug_data["gello_right"].append(
                        components["right/joint_pos"].copy())
                    debug_data["yam_right"].append(
                        obs["robot"]["right/joint_pos"].copy())
                    debug_data["gello_left"].append(
                        components["left/joint_pos"].copy())
                    debug_data["yam_left"].append(
                        obs["robot"]["left/joint_pos"].copy())

                # Stream video frames to ffmpeg (no memory accumulation)
                if not video_writer.is_started:
                    first_frame = next(iter(obs["images"].values()))
                    h, w = first_frame.shape[:2]
                    video_writer.start(list(obs["images"].keys()), w, h)
                for cam_name in obs["images"]:
                    video_writer.write_frame(cam_name, obs["images"][cam_name])

                # Send the action and advance to the next obs. This blocks ~16ms
                # for the next broker frame and is what paces the loop at 60Hz.
                obs, _, _, _, _ = env.step(action)

                step_count = len(episode["actions"])
                audio_state["step"] = step_count
                if step_count <= 5:
                    print(f"  step {step_count}: "
                          f"{time.time()-t0:.3f}s since teleop start")
                if step_count % 60 == 0:
                    print(f"  Recording: {step_count} steps "
                          f"({step_count / 60:.1f}s)")
                if step_count >= args.max_steps:
                    print(f"  Max steps reached ({args.max_steps})")
                    break

                now_t = time.monotonic()
                if now_t - last_diag_t >= 1.0:
                    fps = ((step_count - recorded_at_last_diag) /
                           (now_t - last_diag_t))
                    print(f"  [diag] recorded={step_count} fps={fps:.1f}")
                    last_diag_t = now_t
                    recorded_at_last_diag = step_count

                # --- Pedal / key polling (every iteration) ---
                pedal = hub.poll_press({KEY_FAILURE, KEY_SUCCESS})
                if pedal is not None:
                    episode_outcome = (pedal == KEY_SUCCESS)
                    break

                key = keys.get_key()
                if key == "q":
                    print(">> Quitting (q in main loop)")
                    raise SystemExit("main-loop-q")

                # If neither queue had a message, sleep briefly so we don't
                # busy-spin between gello samples (~5ms apart at 200Hz).
                if gello_msg is None and new_obs is None:
                    time.sleep(0.001)

            # If max_steps hit, wait for pedal to mark outcome
            if episode_outcome is None:
                print("  Press pedal: left=FAILURE, right=SUCCESS")
                while True:
                    pedal = hub.poll_press({KEY_FAILURE, KEY_SUCCESS})
                    if pedal is not None:
                        episode_outcome = (pedal == KEY_SUCCESS)
                        break
                    key = keys.get_key()
                    if key == "q":
                        print(">> Quitting (q in pedal-wait)")
                        raise SystemExit("pedal-wait-q")
                    time.sleep(0.05)

            # === PHASE 4: SAVE EPISODE ===
            audio_state["step"] = -1
            n = len(episode["actions"])
            label = "SUCCESS" if episode_outcome else "FAILURE"
            print(f"\n>> Episode ended: {label} ({n} steps)")

            if debug_data is not None and n > 0:
                plot_debug_episode(debug_data, episode_dir)

            # Finalize video encoding
            print(">> Finalizing video encoding...")
            video_result = video_writer.finish()

            # Move video files into episode dir
            video_info = {}
            for cam_name, (tmp_path, num_frames) in video_result.items():
                safe_name = cam_name.replace("/", "_")
                ext = codec_config.container
                final_name = f"{safe_name}.{ext}"
                final_path = Path(episode_dir) / final_name
                shutil.move(str(tmp_path), str(final_path))
                video_info[cam_name] = (final_name, num_frames)

            # Save HDF5
            hdf5_path = os.path.join(episode_dir, "episode.hdf5")
            print(f">> Saving to {hdf5_path}")
            save_episode(hdf5_path, episode, video_info, codec_config,
                         success=episode_outcome)

            # Clean up temp video dir
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir = None
            video_writer = None

            # Rename failed episodes for easy identification
            if not episode_outcome:
                failed_dir = os.path.join(args.output_dir,
                                          f"FAILED_{episode_name}")
                os.rename(episode_dir, failed_dir)
                episode_dir = failed_dir

            episode_count += 1
            print(f">> Saved: {episode_dir}")
            print(f"   Total episodes: {episode_count}")

            episode = None
            print()  # blank line before next cycle

    except KeyboardInterrupt:
        print(">> Exit cause: KeyboardInterrupt (SIGINT)")
    except SystemExit as e:
        print(f">> Exit cause: SystemExit (code={e.code})")
    except BaseException as e:
        import traceback
        print(f">> Exit cause: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # Ensure GELLO torque is off (prevents 200Hz loop slowdown if left on)
        try:
            gello_cmd.send_json({"command": "disable_torque"})
        except Exception:
            pass
        # Clean up in-progress recording
        if video_writer is not None:
            video_writer.cleanup()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        audio_recorder.stop()
        hub.close()
        keys.close()
        gello_cmd.close()
        gello_cmd_ctx.term()
        env.close()
        print(f"Done. Collected {episode_count} episodes.")


if __name__ == "__main__":
    main()
