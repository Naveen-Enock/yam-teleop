"""Data collection for the YAM leader-follower system.

Successor to collect_data.py for the YAM-leader teleop stack. Differences:
  - Leader = YAM arms with teaching handles (yam_leader_node), not GELLO.
    The :5004 wire contract is identical, so env / broker / video are unchanged
    and the recorded HDF5 schema is the same (the leader stream is still stored
    under the "gello" group for downstream-tool compatibility).
  - Sub-task MARKERS come from the teaching-handle marker button (button 0,
    via the leader's monotonic buttons.marker_seq) OR the audio foot pedal OR
    the 'm' key. Stored as a `subtask_markers` dataset (step indices) in the
    HDF5 — no audio/microphone anymore (that path is deprecated and removed).
  - CLUTCH (handle button 1): while the leader reports mode == "suspended" the
    follower holds and recording pauses; on resume it continues seamlessly.
  - Foot pedals are OPTIONAL. With pedals: right = SUCCESS, left = FAILURE,
    audio pedal = marker. Without pedals (e.g. a dev box): keyboard fallback
    s = SUCCESS, f = FAILURE, m = marker. 'q' quits in both cases.

The old collect_data.py is left untouched for backward compatibility.

Usage:
    python -m yam_teleop.scripts.collect_yam \
        --env-config configs/env.yaml --output-dir data/task_name
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
from yam_teleop.video import (
    DEFAULT_PRESET,
    VIDEO_PRESETS,
    StreamingVideoWriter,
)

# Leader trigger value must be below this to count as "closed" (squeezed).
GRIPPER_CLOSE_THRESHOLD = 0.1


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

    def get_key(self):
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


def both_grippers_closed(leader_msg: dict, threshold: float) -> bool:
    """True when both leader triggers are squeezed past the threshold."""
    left = float(leader_msg["left"]["gripper_pos"])
    right = float(leader_msg["right"]["gripper_pos"])
    return left < threshold and right < threshold


def leader_msg_to_action(leader_msg: dict):
    """Convert a leader state message to (action_vector, components, ts_ns)."""
    left_jp = np.asarray(leader_msg["left"]["joint_pos"])
    left_gp = np.asarray([float(leader_msg["left"]["gripper_pos"])])
    right_jp = np.asarray(leader_msg["right"]["joint_pos"])
    right_gp = np.asarray([float(leader_msg["right"]["gripper_pos"])])
    action = np.concatenate([left_jp, left_gp, right_jp, right_gp])
    components = {
        "left/joint_pos": left_jp,
        "left/gripper_pos": left_gp,
        "right/joint_pos": right_jp,
        "right/gripper_pos": right_gp,
    }
    return action, components, int(leader_msg["timestamp_ns"])


def reset_leader(env, leader_cmd, home_joints_left, home_joints_right,
                 gripper_threshold=0.1, joint_speed=0.2, send_hz=200.0,
                 keys=None):
    """Drive the leader arms to the follower start pose, then wait for squeeze.

    Same handshake the old reset_gello used: enable_torque -> stream waypoints
    to the home pose -> hold until both triggers are squeezed -> disable_torque
    (which the leader node interprets as: RESET stiff-match, then COUPLED).
    """
    leader_cmd.send_json({"command": "enable_torque", "gripper_limp": True})

    leader_msg = None
    deadline = time.monotonic() + 2.0
    while leader_msg is None and time.monotonic() < deadline:
        leader_msg = env.get_latest_gello()
        time.sleep(0.005)
    if leader_msg is None:
        raise RuntimeError("No leader messages received after enable_torque")

    current_left = np.asarray(leader_msg["left"]["joint_pos"])
    current_right = np.asarray(leader_msg["right"]["joint_pos"])
    target_left = np.array(home_joints_left)
    target_right = np.array(home_joints_right)

    max_d = max(np.max(np.abs(current_left - target_left)),
                np.max(np.abs(current_right - target_right)))
    duration = max(max_d / joint_speed, 0.05)
    num_steps = max(int(duration * send_hz), 1)
    print(f"  Leader reset: {num_steps} steps over {duration:.2f}s "
          f"(max_delta={max_d:.3f} rad)")
    period = 1.0 / send_hz
    next_t = time.monotonic()
    for lw, rw in zip(
        np.linspace(current_left, target_left, num_steps + 1)[1:],
        np.linspace(current_right, target_right, num_steps + 1)[1:],
    ):
        leader_cmd.send_json({
            "left": {"joint_pos": lw.tolist(), "gripper_pos": 0.0},
            "right": {"joint_pos": rw.tolist(), "gripper_pos": 0.0},
        })
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.monotonic()

    print("  Leader at neutral. Squeeze both triggers to start...")
    while True:
        leader_cmd.send_json({
            "left": {"joint_pos": target_left.tolist(), "gripper_pos": 0.0},
            "right": {"joint_pos": target_right.tolist(), "gripper_pos": 0.0},
        })
        leader_msg = env.get_latest_gello()
        if leader_msg is not None and both_grippers_closed(
                leader_msg, gripper_threshold):
            break
        time.sleep(0.005)
        if keys is not None and keys.get_key() == "q":
            print(">> Quitting (q in reset_leader)")
            raise SystemExit("reset-q")

    leader_cmd.send_json({"command": "disable_torque"})


def save_episode(path: str, episode: dict, video_info: dict,
                 codec_config, success: bool = False) -> None:
    """Save one episode to HDF5 with video file references.

    Schema matches collect_data.py (leader stream under the "gello" group for
    downstream-tool compatibility) plus a `subtask_markers` dataset.
    """
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

        # Sub-task boundary markers: step indices at which the operator pressed
        # the marker button/pedal/key. Replaces the old audio-metadata markers.
        f.create_dataset(
            "subtask_markers",
            data=np.array(episode["subtask_markers"], dtype=np.int64),
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
        f.attrs["obs_action_aligned"] = True


def main():
    parser = argparse.ArgumentParser(description="Collect YAM leader-follower teleop data")
    parser.add_argument("--env-config", required=True, help="Path to env.yaml")
    parser.add_argument("--output-dir", required=True, help="Directory to save episodes")
    parser.add_argument("--max-steps", type=int, default=18000,
                        help="Max steps per episode (default: 18000 = 5min at 60Hz)")
    parser.add_argument("--gripper-threshold", type=float, default=GRIPPER_CLOSE_THRESHOLD,
                        help="Leader trigger close threshold (default: 0.1)")
    parser.add_argument("--leader-cmd-port", type=int, default=5006,
                        help="ZMQ port for leader commands (default: 5006)")
    parser.add_argument("--video-codec", default=DEFAULT_PRESET,
                        choices=list(VIDEO_PRESETS.keys()),
                        help=f"Video codec preset (default: {DEFAULT_PRESET})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    task_name = Path(args.output_dir).name
    codec_config = VIDEO_PRESETS[args.video_codec]

    env = YAMBimanualEnv(args.env_config)

    # Leader command socket (PUSH -> yam_leader_node PULL)
    leader_ctx = zmq.Context()
    leader_cmd = leader_ctx.socket(zmq.PUSH)
    leader_cmd.setsockopt(zmq.LINGER, 0)
    leader_cmd.connect(f"tcp://127.0.0.1:{args.leader_cmd_port}")
    time.sleep(0.5)

    # Foot pedals are optional — fall back to keyboard if none are attached.
    try:
        hub = FootPedalHub()
    except RuntimeError as e:
        hub = None
        print(f"[pedals] none found ({e}); using keyboard: "
              "s=SUCCESS f=FAILURE m=MARKER q=QUIT")

    keys = KeyListener()
    episode_count = 0
    video_writer = None
    tmp_dir = None

    print("\nData Collection Cycle (YAM leader-follower):")
    print("  1. Follower reset to start pose (grippers open then close)")
    print("  2. Leader matches follower; squeeze both triggers to start")
    print("  3. Teleop + recording")
    print("  4. Marker: handle button 0 / audio pedal / 'm'")
    print("  5. Clutch: handle button 1 (pause/resume)")
    print("  6. End: right pedal / 's' = SUCCESS, left pedal / 'f' = FAILURE")
    print(f"  Trigger close threshold: {args.gripper_threshold}")
    print(f"  Leader cmd port: {args.leader_cmd_port}")
    print(f"  Video codec: {args.video_codec}\n")

    def poll_outcome(key):
        """Return True (success) / False (failure) / None from pedal or key."""
        if hub is not None:
            pedal = hub.poll_press({KEY_FAILURE, KEY_SUCCESS})
            if pedal is not None:
                return pedal == KEY_SUCCESS
        if key == "s":
            return True
        if key == "f":
            return False
        return None

    try:
        while True:
            # === PHASE 1: RESET FOLLOWER ===
            obs, info = env.reset()

            # === PHASE 2: MATCH LEADER ===
            print("Resetting leader (match follower, squeeze both triggers)...")
            reset_leader(
                env, leader_cmd,
                home_joints_left=env._home_position[:6].tolist(),
                home_joints_right=env._home_position[7:13].tolist(),
                gripper_threshold=args.gripper_threshold,
                keys=keys,
            )
            print(">> Leader ready")

            # === PHASE 3: TELEOP + RECORD ===
            t0 = time.time()
            leader_msg = None
            deadline = time.monotonic() + 5.0
            while leader_msg is None and time.monotonic() < deadline:
                leader_msg = env.get_latest_gello()
                if leader_msg is None:
                    time.sleep(0.005)
            if leader_msg is None:
                raise RuntimeError("No leader messages within 5s of teleop start")
            print(f">> First leader msg received ({time.time()-t0:.3f}s)")

            warmup_obs = env.poll_broker_obs(timeout_ms=2000)
            if warmup_obs is None:
                raise RuntimeError("No broker obs within 2s of teleop start")
            print(">> Teleop + recording started")

            episode_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"{task_name}_{episode_timestamp}"
            episode_dir = os.path.join(args.output_dir, episode_name)
            os.makedirs(episode_dir, exist_ok=True)

            tmp_dir = tempfile.mkdtemp(dir=episode_dir, prefix=".tmp_video_")
            video_writer = StreamingVideoWriter(tmp_dir, codec=codec_config, fps=60)

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
                "subtask_markers": [],
            }

            if hub is not None:
                hub.clear_pending()  # discard stale pedal presses from reset
            episode_outcome = None
            was_suspended = False
            last_marker_seq = (leader_msg.get("buttons") or {}).get("marker_seq", 0)

            t0 = time.time()
            last_diag_t = time.monotonic()
            recorded_at_last_diag = 0
            obs = warmup_obs
            while True:
                fresh = env.get_latest_gello()
                if fresh is not None:
                    leader_msg = fresh
                buttons = leader_msg.get("buttons") or {}
                mode = leader_msg.get("mode", "coupled")
                key = keys.get_key()

                # --- CLUTCH: pause recording + commanding while suspended ---
                if mode == "suspended":
                    if not was_suspended:
                        print("  >> CLUTCH: paused (follower holding)")
                        was_suspended = True
                    outcome = poll_outcome(key)
                    if outcome is not None:
                        episode_outcome = outcome
                        break
                    if key == "q":
                        raise SystemExit("paused-q")
                    time.sleep(0.01)
                    continue
                if was_suspended:
                    print("  >> CLUTCH: resumed")
                    was_suspended = False
                    fresh_obs = env.poll_broker_obs(timeout_ms=500)
                    if fresh_obs is not None:
                        obs = fresh_obs

                # --- MARKER (handle button seq / audio pedal / 'm' key) ---
                marker_seq = buttons.get("marker_seq", last_marker_seq)
                marker_hit = marker_seq != last_marker_seq
                last_marker_seq = marker_seq
                if hub is not None and hub.poll_press({KEY_AUDIO}):
                    marker_hit = True
                if key == "m":
                    marker_hit = True
                if marker_hit:
                    episode["subtask_markers"].append(len(episode["actions"]))
                    print(f"  >> MARKER at step {len(episode['actions'])}")

                action, components, gello_ns = leader_msg_to_action(leader_msg)

                episode["actions"].append(action.copy())
                episode["camera_ns"].append(obs["timestamps"]["camera_ns"])
                episode["robot_ns"].append(obs["timestamps"]["robot_ns"])
                episode["gello_ns"].append(gello_ns)
                episode["broker_ns"].append(obs["timestamps"]["broker_ns"])
                for k in episode["robot"]:
                    episode["robot"][k].append(obs["robot"][k].copy())
                for k in gello_record_keys:
                    episode["gello"][k].append(components[k].copy())

                if not video_writer.is_started:
                    first_frame = next(iter(obs["images"].values()))
                    h, w = first_frame.shape[:2]
                    video_writer.start(list(obs["images"].keys()), w, h)
                for cam_name in obs["images"]:
                    video_writer.write_frame(cam_name, obs["images"][cam_name])

                obs, _, _, _, _ = env.step(action)

                step_count = len(episode["actions"])
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
                    print(f"  [diag] recorded={step_count} fps={fps:.1f} "
                          f"markers={len(episode['subtask_markers'])}")
                    last_diag_t = now_t
                    recorded_at_last_diag = step_count

                outcome = poll_outcome(key)
                if outcome is not None:
                    episode_outcome = outcome
                    break
                if key == "q":
                    raise SystemExit("main-loop-q")

            if episode_outcome is None:
                print("  End episode: right/'s'=SUCCESS, left/'f'=FAILURE")
                while True:
                    key = keys.get_key()
                    outcome = poll_outcome(key)
                    if outcome is not None:
                        episode_outcome = outcome
                        break
                    if key == "q":
                        raise SystemExit("pedal-wait-q")
                    time.sleep(0.05)

            # === PHASE 4: SAVE ===
            n = len(episode["actions"])
            label = "SUCCESS" if episode_outcome else "FAILURE"
            print(f"\n>> Episode ended: {label} ({n} steps, "
                  f"{len(episode['subtask_markers'])} markers)")

            print(">> Finalizing video encoding...")
            video_result = video_writer.finish()
            video_info = {}
            for cam_name, (tmp_path, num_frames) in video_result.items():
                safe_name = cam_name.replace("/", "_")
                final_name = f"{safe_name}.{codec_config.container}"
                final_path = Path(episode_dir) / final_name
                shutil.move(str(tmp_path), str(final_path))
                video_info[cam_name] = (final_name, num_frames)

            hdf5_path = os.path.join(episode_dir, "episode.hdf5")
            print(f">> Saving to {hdf5_path}")
            save_episode(hdf5_path, episode, video_info, codec_config,
                         success=episode_outcome)

            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir = None
            video_writer = None

            if not episode_outcome:
                failed_dir = os.path.join(args.output_dir, f"FAILED_{episode_name}")
                os.rename(episode_dir, failed_dir)
                episode_dir = failed_dir

            episode_count += 1
            print(f">> Saved: {episode_dir}")
            print(f"   Total episodes: {episode_count}\n")
            episode = None

    except KeyboardInterrupt:
        print(">> Exit cause: KeyboardInterrupt (SIGINT)")
    except SystemExit as e:
        print(f">> Exit cause: SystemExit (code={e.code})")
    except BaseException as e:
        import traceback
        print(f">> Exit cause: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            leader_cmd.send_json({"command": "disable_torque"})
        except Exception:
            pass
        if video_writer is not None:
            video_writer.cleanup()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if hub is not None:
            hub.close()
        keys.close()
        leader_cmd.close()
        leader_ctx.term()
        env.close()
        print(f"Done. Collected {episode_count} episodes.")


if __name__ == "__main__":
    main()
