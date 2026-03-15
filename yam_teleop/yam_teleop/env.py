"""YAMBimanualEnv: gym.Env wrapping the sync broker for bimanual YAM teleop.

Subscribes to the sync broker for observations, publishes joint commands to the
robot node. Used identically for data collection (action from GELLO) and policy
inference (action from neural network).
"""

import argparse
import collections
import threading
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
import zmq


class YAMBimanualEnv(gym.Env):
    """Bimanual YAM robot environment.

    Observation: dict with "images", "robot", "gello", "timestamps" keys.
    Action: 14-dim array [left_arm(6), left_grip(1), right_arm(6), right_grip(1)].
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str):
        super().__init__()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._broker_port = cfg["broker_port"]
        self._cmd_port = cfg["robot_cmd_port"]
        # Direct gello state stream (bypasses broker). When set, callers can use
        # get_latest_gello() to drive YAM commands at gello's native rate
        # instead of being gated on the camera-rate broker.
        self._gello_state_port = cfg.get("gello_state_port", 5004)
        # Direct camera image stream (bypasses broker). The broker only sends
        # synchronized metadata (cam_timestamp + robot_state). Image bytes
        # come directly from camera_node and we pair them by timestamp_ns.
        self._camera_port = cfg.get("camera_port", 5001)
        self._home_position = np.array(cfg["home_position"], dtype=np.float64)
        self._reset_tol = cfg["reset_tolerance"]
        self._control_freq = cfg["control_freq_hz"]
        self._control_period = 1.0 / self._control_freq
        # Max joint delta per step during reset (rad). Limits arm speed.
        self._reset_max_delta = cfg.get("reset_max_delta", 0.01)
        # Gripper open limit in command space
        self._gripper_max = cfg.get("gripper_max_open", 0.85)
        # Folded "rest"/sit pose the arms park in on shutdown (all joints 0 ->
        # arms fold down at the base). Guarded: defaults to the safe-shutdown
        # pose (mirrors robot.yaml) when the config omits `rest_position`.
        self._rest_position = np.array(
            cfg.get("rest_position", [0.0] * 6 + [self._gripper_max] + [0.0] * 6 + [self._gripper_max]),
            dtype=np.float64,
        )

        # Action space: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64,
        )

        # Observation space is a dict — we define it loosely since shapes depend
        # on camera resolution (configured elsewhere).
        self.observation_space = gym.spaces.Dict({
            "robot": gym.spaces.Dict({
                "left/joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "left/joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "left/joint_eff": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "left/gripper_pos": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                "left/gripper_eff": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                "right/joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "right/joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "right/joint_eff": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "right/gripper_pos": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                "right/gripper_eff": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
            }),
        })
        # gello is no longer part of the broker obs — clients should call
        # get_latest_gello() directly to read the live (sub-5ms) gello state.

        # ZMQ setup
        self._ctx = zmq.Context()

        # CONFLATE=1 keeps only the latest broker meta — older queued metas
        # are dropped on arrival. RCVHWM=1 with default drop semantics is
        # FIFO-first-stays, which causes env to consume stale metas while
        # the camera stream advances (the streams never converge in the
        # _get_obs pairing loop). Conflate semantics match _gello_sub.
        self._obs_sub = self._ctx.socket(zmq.SUB)
        self._obs_sub.setsockopt(zmq.CONFLATE, 1)
        self._obs_sub.connect(f"tcp://127.0.0.1:{self._broker_port}")
        self._obs_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self._cmd_pub = self._ctx.socket(zmq.PUB)
        self._cmd_pub.setsockopt(zmq.SNDHWM, 1)
        self._cmd_pub.bind(f"tcp://127.0.0.1:{self._cmd_port}")

        # Direct subscription to gello_node state. CONFLATE=1 keeps only the
        # most recent message — callers always see the freshest gello state
        # without queueing, regardless of camera/broker timing.
        self._gello_sub = self._ctx.socket(zmq.SUB)
        self._gello_sub.setsockopt(zmq.CONFLATE, 1)
        self._gello_sub.connect(f"tcp://127.0.0.1:{self._gello_state_port}")
        self._gello_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Direct subscription to the camera image stream. Image bytes go
        # camera_node → here without passing through the broker. Each frame
        # is paired with the broker's sync metadata by timestamp_ns (set
        # once by camera_node, identical on both streams).
        #
        # Subtle: ZMQ SUB drops *new* arrivals when its queue is full, so
        # the queue head never advances past whatever frames sat there when
        # the queue first filled — we'd be perpetually 16+ seconds behind.
        # CONFLATE would fix it but doesn't support multipart messages.
        # Workaround: a background thread continuously drains the socket
        # and stashes only the latest (cam_meta, image_parts) in memory.
        self._camera_sub = self._ctx.socket(zmq.SUB)
        self._camera_sub.setsockopt(zmq.RCVHWM, 8)
        self._camera_sub.connect(f"tcp://127.0.0.1:{self._camera_port}")
        self._camera_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._cam_lock = threading.Lock()
        # Recent camera frames keyed by timestamp_ns. Sized so the matching
        # frame for any given broker meta is reliably in-deque by the time
        # env._get_obs goes looking for it (drain is ~1 frame interval
        # behind broker because image bytes take ~17ms to copy at 60Hz).
        self._cam_buffer: collections.deque[Tuple[Dict[str, Any], list]] = (
            collections.deque(maxlen=8))
        self._cam_thread_stop = threading.Event()
        self._cam_thread = threading.Thread(
            target=self._drain_camera, name="cam-drain", daemon=True)
        self._cam_thread.start()

        # Poller for non-blocking broker obs reads from teleop loops that
        # want to send commands faster than the broker publishes.
        self._obs_poller = zmq.Poller()
        self._obs_poller.register(self._obs_sub, zmq.POLLIN)

        time.sleep(1.0)  # let connections establish

        # State
        self._step_count = 0
        self._last_obs = None

    def _drain_camera(self) -> None:
        """Background thread: drain camera_sub into the recent-frames deque."""
        while not self._cam_thread_stop.is_set():
            try:
                cam_meta = self._camera_sub.recv_json()
                n = len(cam_meta["cameras"])
                parts = [self._camera_sub.recv() for _ in range(n)]
            except zmq.ContextTerminated:
                return
            except Exception:
                if self._cam_thread_stop.is_set():
                    return
                continue
            with self._cam_lock:
                self._cam_buffer.append((cam_meta, parts))

    def _find_camera_by_ts(self, target_ts: int) -> Optional[
            Tuple[Dict[str, Any], list]]:
        """Look up the buffered frame with `timestamp_ns == target_ts`."""
        with self._cam_lock:
            for entry in reversed(self._cam_buffer):
                if entry[0]["timestamp_ns"] == target_ts:
                    return entry
        return None

    def _get_obs(self) -> Dict[str, Any]:
        """Receive one synchronized observation.

        Reads the latest broker meta (CONFLATE) and waits for the camera
        frame with the same `timestamp_ns` to be fully drained. The drain
        thread is always one frame interval behind broker because image
        bytes take ~17ms to deliver at 1920×1080 × 3 cams at 60Hz.

        Tolerates stale broker metas. After a multi-second pause (e.g.
        save_episode finalising ffmpeg + HDF5), the broker meta sitting
        in our CONFLATE slot can predate every frame in the drain deque
        (maxlen ≈ 133 ms of history), making its matching frame
        permanently unrecoverable. We detect that case and drop the meta
        instead of timing out, then re-recv until we get one whose frame
        is still in-deque or imminent.
        """
        t0 = time.monotonic()
        deadline = t0 + 1.0

        while True:
            meta = self._obs_sub.recv_json()
            expected_ts = meta["timestamps"]["camera_ns"]

            cam = self._find_camera_by_ts(expected_ts)
            while cam is None:
                # If the deque's oldest frame is already newer than
                # expected_ts, this meta is unrecoverably stale — its
                # matching frame has been evicted. Drop it and re-recv.
                with self._cam_lock:
                    oldest_cam_ts = (
                        self._cam_buffer[0][0]["timestamp_ns"]
                        if self._cam_buffer else None
                    )
                if oldest_cam_ts is not None and expected_ts < oldest_cam_ts:
                    break  # drop meta, retry outer

                if time.monotonic() > deadline:
                    with self._cam_lock:
                        seen = [e[0]["timestamp_ns"] for e in self._cam_buffer]
                    deltas = [s - expected_ts for s in seen]
                    raise RuntimeError(
                        f"_get_obs: no camera frame with ts={expected_ts} "
                        f"appeared in drain buffer within 1s. "
                        f"buffer ts deltas (ns): {deltas}")
                time.sleep(0.001)
                cam = self._find_camera_by_ts(expected_ts)

            if cam is not None:
                break  # found a match — fall through to build obs

        camera_meta = meta["camera_meta"]
        camera_names = camera_meta["cameras"]
        height = camera_meta["height"]
        width = camera_meta["width"]
        channels = camera_meta["channels"]

        cam_meta, parts = cam
        images = {
            name: np.frombuffer(parts[i], dtype=np.uint8).reshape(
                height, width, channels)
            for i, name in enumerate(camera_names)
        }

        t_pair = time.monotonic() - t0
        if t_pair > 0.05:
            import sys
            print(f"  [env._get_obs] t_pair={t_pair*1000:.1f}ms",
                  file=sys.stderr)

        robot = meta["robot"]
        timestamps = meta["timestamps"]

        obs = {
            "images": images,
            "robot": {
                "left/joint_pos": np.array(robot["left"]["joint_pos"]),
                "left/joint_vel": np.array(robot["left"]["joint_vel"]),
                "left/joint_eff": np.array(robot["left"]["joint_eff"]),
                "left/gripper_pos": np.array([robot["left"]["gripper_pos"]]),
                "left/gripper_eff": np.array([robot["left"]["gripper_eff"]]),
                "right/joint_pos": np.array(robot["right"]["joint_pos"]),
                "right/joint_vel": np.array(robot["right"]["joint_vel"]),
                "right/joint_eff": np.array(robot["right"]["joint_eff"]),
                "right/gripper_pos": np.array([robot["right"]["gripper_pos"]]),
                "right/gripper_eff": np.array([robot["right"]["gripper_eff"]]),
            },
            "timestamps": timestamps,
        }
        self._last_obs = obs
        return obs

    def get_latest_gello(self) -> Optional[Dict[str, Any]]:
        """Non-blocking read of the latest gello_node state message.

        Bypasses the broker entirely — returns the freshest gello sample
        published on gello_state_port (CONFLATE=1, so old samples are dropped).
        Returns None if no message has been received yet.

        Use this in teleop loops that want to drive YAM at gello's native rate
        (~200Hz) instead of being gated on the broker's camera rate (~60Hz).
        """
        try:
            return self._gello_sub.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None

    def poll_broker_obs(self, timeout_ms: int = 0) -> Optional[Dict[str, Any]]:
        """Non-blocking variant of _get_obs.

        Returns the next synchronized broker observation if one is ready
        within timeout_ms (default: don't wait at all), else None. Used by
        teleop loops that want to send commands faster than the broker
        publishes but still record one frame per broker cycle.
        """
        socks = dict(self._obs_poller.poll(timeout=timeout_ms))
        if self._obs_sub not in socks:
            return None
        return self._get_obs()

    def send_command(self, action: np.ndarray) -> None:
        """Public command publisher. Bypasses step()'s broker-blocking obs read."""
        self._send_command(action)

    def _send_command(self, action: np.ndarray) -> None:
        """Send joint position command to robot node."""
        msg = {
            "timestamp_ns": time.time_ns(),
            "left": {
                "joint_pos": action[:6].tolist(),
                "gripper_pos": float(action[6]),
            },
            "right": {
                "joint_pos": action[7:13].tolist(),
                "gripper_pos": float(action[13]),
            },
        }
        self._cmd_pub.send_json(msg)

    def _get_current_joints(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract the 14-dim joint vector from an observation."""
        return np.concatenate([
            obs["robot"]["left/joint_pos"],
            obs["robot"]["left/gripper_pos"],
            obs["robot"]["right/joint_pos"],
            obs["robot"]["right/gripper_pos"],
        ])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        # Get current position
        obs = self._get_obs()
        current = self._get_current_joints(obs)
        target = self._home_position

        # Phase 1: move to home with grippers open
        target_open = target.copy()
        target_open[6] = self._gripper_max   # left gripper open
        target_open[13] = self._gripper_max  # right gripper open

        self._move_to(current, target_open, "Resetting (grippers open)...")
        self._hold_until_converged(target_open)

        # Phase 2: close grippers
        target_closed = target.copy()
        target_closed[6] = 0.0   # left gripper closed
        target_closed[13] = 0.0  # right gripper closed

        current = self._get_current_joints(self._get_obs())
        self._move_to(current, target_closed, "Closing grippers...")
        self._hold_until_converged(target_closed)

        obs = self._get_obs()
        return obs, {}

    def return_to_rest(self) -> None:
        """Gently move both arms to the folded rest/sit pose at the base.

        Used on shutdown so the arms park low instead of holding the raised
        reset()/working pose. Same gradual interpolation as reset(); no gym
        bookkeeping since this is a teardown move, not an episode reset.
        """
        current = self._get_current_joints(self._get_obs())
        self._move_to(current, self._rest_position, "Returning to rest pose...")
        self._hold_until_converged(self._rest_position)

    def _move_to(self, current: np.ndarray, target: np.ndarray,
                 msg: str) -> None:
        """Gradual linear interpolation from current to target."""
        max_delta = np.max(np.abs(current - target))
        if max_delta < self._reset_tol:
            return
        num_steps = max(int(max_delta / self._reset_max_delta), 1)
        print(f"  {msg} ({num_steps} steps, max_delta={max_delta:.3f} rad)")
        waypoints = np.linspace(current, target, num_steps + 1)[1:]
        for waypoint in waypoints:
            self._send_command(waypoint)
            self._get_obs()

    def _hold_until_converged(self, target: np.ndarray,
                              max_iters: int = 200) -> None:
        """Hold target position until convergence."""
        for i in range(max_iters):
            self._send_command(target)
            obs = self._get_obs()
            current = self._get_current_joints(obs)
            error = np.max(np.abs(current - target))
            if error < self._reset_tol:
                print(f"  Converged (max_error={error:.4f})")
                return
        print(f"  Warning: did not fully converge (max_error={error:.4f})")

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(14)
        self._send_command(action)

        obs = self._get_obs()  # blocks until next broker frame (~60Hz)
        self._step_count += 1

        reward = 0.0
        terminated = False
        truncated = False
        info = {"step": self._step_count}

        return obs, reward, terminated, truncated, info

    def close(self):
        # Signal the drain thread first; closing the socket out from under
        # it would raise inside the recv.
        self._cam_thread_stop.set()
        if self._obs_sub is not None:
            self._obs_sub.close()
            self._obs_sub = None
        if self._gello_sub is not None:
            self._gello_sub.close()
            self._gello_sub = None
        if self._camera_sub is not None:
            self._camera_sub.close()
            self._camera_sub = None
        if self._cmd_pub is not None:
            self._cmd_pub.close()
            self._cmd_pub = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
