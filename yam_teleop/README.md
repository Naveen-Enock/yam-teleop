# YAM Bimanual Teleop System

A decoupled microservice system for bimanual robot teleoperation, data
collection, and policy inference using YAM arms and GELLO leader devices.

## Hardware

- **2x YAM robot arms** — 6-DOF + gripper each, CAN bus control via `i2rt`
  library. Internal 250Hz gravity-compensated control loop.
- **2x GELLO leader arms** — Dynamixel servo-based arms used for
  teleoperation. 6 joints + gripper trigger each. Passive (torque-off)
  during teleop; the gello_node can also drive them (torque-on, extended
  position control) — used during reset to bring them to home pose.
- **3x USB webcams** — top, left wrist, right wrist. 1920x1080 @ 60Hz,
  MJPG raw passthrough, manual exposure/white balance.
- **2x USB foot pedals** — single-button audio/marker pedal and
  2-button success/failure pedal (PCsensor FootSwitch, vendor 0x3553).

## Architecture

Each hardware component runs as an independent process. They communicate over
ZMQ. The broker synchronizes robot state to camera timestamps; image bytes and
GELLO state bypass the broker and flow directly to consumers. A standard
gym.Env interface sits on top, used identically for data collection and policy
inference.

```
camera_node                 robot_node              gello_node
  3x USB webcams              2x YAM arms             2x GELLO arms
  PUB :5001 images @60Hz      PUB :5002 @200Hz        PUB :5004 @200Hz
  PUB :5008 meta   @60Hz      SUB :5003 cmds          PULL :5006 cmds
   |        \                       |                       |
   |         \                      v                       |
   |          +---> sync_broker                             |
   |                  SUB :5008 (camera meta — JSON only)   |
   |                  SUB :5002 (robot)                     |
   |                  interpolates robot at camera ts       |
   |                  PUB :5005 (JSON only — no images)     |
   |                       |                                |
   v                       v                                v
   +-----------------------+--------------------------------+
                           |
                           v
                  YAMBimanualEnv (gym.Env)
                    SUB :5005 broker meta (CONFLATE)
                    SUB :5001 images (background drain thread)
                    SUB :5004 gello   (CONFLATE, get_latest_gello())
                    PUB :5003 YAM commands
                           |
                  +--------+--------+
                  |                 |
            collect_data.py    run_policy.py
```

### ZMQ Ports

| Port | From → To           | Type     | Content |
|------|---------------------|----------|---------|
| 5001 | camera_node → env   | PUB/SUB  | Multipart: JSON metadata + 3 raw BGR image buffers |
| 5002 | robot_node → broker | PUB/SUB  | JSON: left/right joint_pos, joint_vel, joint_eff, gripper_pos, gripper_eff |
| 5003 | env → robot_node    | PUB/SUB  | JSON: left/right joint position commands |
| 5004 | gello_node → env    | PUB/SUB  | JSON: left/right joint_pos, gripper_pos, torque_enabled |
| 5005 | broker → env        | PUB/SUB  | JSON: synchronized robot state + timestamps + camera_meta (no images) |
| 5006 | clients → gello_node| PUSH/PULL| JSON: position targets, or `{"command": "enable_torque"|"disable_torque"}` |
| 5008 | camera_node → broker| PUB/SUB  | JSON: camera metadata only (timestamp_ns, names, dims) — no image bytes |

### Synchronization

The broker is **camera-driven**: it blocks on each camera-metadata frame
(60Hz, port 5008), then interpolates the robot state to match the camera
timestamp using a timestamped ring buffer with linear interpolation. The
synchronized output (port 5005) is JSON only; image bytes flow directly from
camera_node to env on port 5001 and are paired with broker meta by exact
`timestamp_ns` equality. GELLO state bypasses the broker entirely — clients
read it directly via `env.get_latest_gello()` for sub-5ms freshness.

If the robot stream becomes stale (>50ms), the broker halts and waits for
recovery before resuming.

## Observation Space

The synchronized observation returned by `env.step()`, `env.reset()`, or
`env.poll_broker_obs()`:

```python
obs = {
    "images": {
        "top":          np.ndarray,  # (1080, 1920, 3) uint8 BGR
        "left_wrist":   np.ndarray,  # (1080, 1920, 3) uint8 BGR
        "right_wrist":  np.ndarray,  # (1080, 1920, 3) uint8 BGR
    },
    "robot": {
        "left/joint_pos":    np.ndarray,  # (6,) radians
        "left/joint_vel":    np.ndarray,  # (6,) rad/s
        "left/joint_eff":    np.ndarray,  # (6,) effort
        "left/gripper_pos":  np.ndarray,  # (1,) 0=closed, 0.85=open
        "left/gripper_eff":  np.ndarray,  # (1,) gripper effort
        "right/joint_pos":   np.ndarray,  # (6,) radians
        "right/joint_vel":   np.ndarray,  # (6,) rad/s
        "right/joint_eff":   np.ndarray,  # (6,) effort
        "right/gripper_pos": np.ndarray,  # (1,) 0=closed, 0.85=open
        "right/gripper_eff": np.ndarray,  # (1,) gripper effort
    },
    "timestamps": {
        "camera_ns":  int,
        "robot_ns":   int,
        "broker_ns":  int,
    },
}
```

GELLO state is **not** in the synchronized obs. Read it directly via
`env.get_latest_gello()`, which returns the freshest gello_node message
(non-blocking, CONFLATE):

```python
gello = env.get_latest_gello()  # or None if no message yet
# gello = {
#   "timestamp_ns": int,
#   "left":  {"joint_pos": [6 floats], "gripper_pos": float},
#   "right": {"joint_pos": [6 floats], "gripper_pos": float},
#   "torque_enabled": bool,
# }
```

GELLO `gripper_pos`: `0` when the operator squeezes the trigger (→ YAM gripper
closes, since YAM `0`=closed), `1` when relaxed (→ YAM clamps to `0.85`=open).

## Action Space

14-dim joint positions: `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`

All values are raw (radians for joints, 0-0.85 for grippers). No normalization.

## Setup

### Prerequisites

- Conda environment `gello` with Python 3.11
- `i2rt` and `gello` packages installed in editable mode from this repo

### Install

The `gello_software` and `i2rt` dependencies are vendored as git submodules
under `third_party/`. From the repo root:

```bash
git submodule update --init --recursive

# Vendored hardware dependencies (editable)
pip install -e third_party/gello_software
pip install -e third_party/gello_software/third_party/DynamixelSDK/python
pip install -e third_party/i2rt

# This package
pip install -e yam_teleop
```

## Running

Each component is launched separately in its own terminal. All commands run
from the `yam_teleop/` directory.

### 1. Camera Node

```bash
python -m yam_teleop.nodes.camera_node --config configs/camera.yaml
```

Opens 3 USB cameras, publishes frames at 60Hz, shows a live preview window.
Use `--no-display` for headless operation.

### 2. GELLO Node

```bash
python -m yam_teleop.nodes.gello_node --config configs/gello.yaml
```

Connects to both GELLO Dynamixel arms, publishes joint state on `:5004` at
200Hz, and accepts commands on `:5006` (PULL): position targets and
`enable_torque` / `disable_torque`. Torque-enable switches both arms to
extended position control mode and recalibrates per-joint offsets.
Add `--debug` to print joint values every 5 seconds.

### 3. Robot Node

```bash
python -m yam_teleop.nodes.robot_node --config configs/robot.yaml
```

Initializes both YAM arms with gravity compensation. Publishes joint state at
200Hz and listens for joint commands. Add `--debug` to print joint values.

On Ctrl+C: gradually returns arms to home position (grippers open), then
shuts down motors. Press Ctrl+C again during the return to stop immediately.

### 4. Sync Broker

```bash
python -m yam_teleop.nodes.sync_broker --config configs/broker.yaml
```

Subscribes to camera-metadata (`:5008`) and robot state (`:5002`), interpolates
the robot state at each camera timestamp (60Hz), and publishes JSON-only
synchronized meta on `:5005`. Image bytes and GELLO state do NOT pass through
the broker — clients consume them directly. Add `--debug` for interpolation
details.

### 5. Data Collection

```bash
python -m yam_teleop.scripts.collect_data \
    --env-config configs/env.yaml \
    --output-dir data/task_name
```

#### Collection Cycle

1. **Reset YAM** — Arms move to home position. Grippers open, then close.
2. **Reset GELLO** — GELLO arms drive (under torque) to the same home pose,
   then go limp. System waits for the operator to squeeze both GELLO triggers
   (`gripper_pos < 0.1`).
3. **Teleop + Record** — Single-rate ~60Hz `env.step(action)` loop:
    - Each iteration: poll the latest GELLO state (CONFLATE keeps the
      freshest 200Hz sample), build the 14-d action, call `env.step(action)`.
    - `env.step` publishes the command to `robot_node` (which CONFLATEs on
      its receive side and re-applies the latest target to YAM at ~250Hz
      internally — so 60Hz publish from here is plenty for smooth motion)
      and blocks for the next synchronized broker obs.
    - Returned obs becomes one recorded frame + one streamed video frame
      to ffmpeg.
    - Same `env.step()` is used by `run_policy.py` and `replay_episode.py`
      for autonomous control — the gym contract is shared across data
      collection and policy evaluation.
4. **End** — Press the **right** foot pedal to save as SUCCESS, the **left**
   pedal to save as FAILURE. Failed-episode directories are renamed with a
   `FAILED_` prefix. Then the cycle returns to step 1.
5. **Audio** — Audio is recorded continuously for the full duration of
   each episode (one WAV + JSON metadata per episode). Tap the audio pedal
   at any sub-task transition to register a `{step, wall_time}` marker
   inside the episode's audio metadata; markers are used downstream to
   slice the per-episode transcript into per-sub-task annotations.

   *Operator tip — clean delivery for clean auto-segmentation.* Pedal
   markers are treated as hard ground truth by `transcribe_audio.py`
   (no snap-to-silence, no punctuation alignment). To avoid leaking
   words across boundaries, end each sub-task phrase cleanly
   (`"…pick up the mug."`), pause, press the pedal, *then* start
   the next phrase fresh (`"Put the mug on the shelf."`).
   The common pitfall is trailing the connecting verb across the
   press (`"…the mug, put"` → press → `"it on the shelf…"`): `put`
   is uttered before the press and gets sliced into the prior
   segment. Residual leaks like this are fixed in the
   post-hoc review, but clean delivery up front means less
   manual cleanup.

   The recorder watches the per-chunk peak amplitude in a rolling 5s
   window and prints a one-shot `[Audio] !!! WARNING: no audio signal…`
   if the mic dies (wireless mic battery, mute, unplug) so you don't
   keep recording silent episodes. Each episode's JSON also stores
   `audio_peak` and `audio_silent`; transcription auto-skips silent
   episodes to avoid Whisper noise-floor hallucinations.
6. **Quit** — Press `q` at any time to exit.

Each episode is saved as a directory containing one HDF5 file plus per-camera
mp4 files (streamed during recording, no in-memory accumulation):

```
data/<task>/<task>_20260314_120000/        # FAILED_<task>_... if failed
├── episode.hdf5
│   ├── actions                   (T, 14) float64
│   ├── timestamps/
│   │   ├── camera_ns             (T,) int64
│   │   ├── robot_ns              (T,) int64
│   │   ├── gello_ns              (T,) int64
│   │   └── broker_ns             (T,) int64
│   ├── robot/
│   │   ├── left/joint_pos        (T, 6) float64
│   │   ├── left/joint_vel        (T, 6) float64
│   │   ├── left/joint_eff        (T, 6) float64
│   │   ├── left/gripper_pos      (T, 1) float64
│   │   ├── left/gripper_eff      (T, 1) float64
│   │   └── right/* (same)        ...
│   ├── gello/
│   │   ├── left/joint_pos        (T, 6) float64
│   │   ├── left/gripper_pos      (T, 1) float64
│   │   ├── right/joint_pos       (T, 6) float64
│   │   └── right/gripper_pos     (T, 1) float64
│   ├── images/                   (groups, no datasets — frames live in mp4)
│   │   ├── top/                  attrs: video_file, codec, num_frames, fps
│   │   ├── left_wrist/           ...
│   │   └── right_wrist/          ...
│   └── attrs: num_steps, success, image_storage="video", created_at
├── top.mp4
├── left_wrist.mp4
├── right_wrist.mp4
├── audio_*.wav                   (one per episode; full-episode mic capture)
└── audio_*.json                  (audio metadata: episode, start/end step,
                                   markers=[{step, wall_time}, ...])
```

### 6. Audio Transcription

```bash
python -m yam_teleop.scripts.transcribe_audio data/task_name
```

Transcribes per-episode audio in a dataset using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3 model,
runs on GPU) with word-level timestamps. The per-episode transcript is
sliced at each marker (and at episode start/end) so each sub-task segment
becomes one annotation in `language_annotations.json`:

```json
[
  {"text": "pick up the mug", "start_step": 114, "end_step": 281, ...},
  {"text": "place it on the shelf",      "start_step": 739, "end_step": 862, ...}
]
```

Options:
- `--model small` — use a smaller/faster model (default: `large-v3`)
- `--device cpu` — run on CPU instead of GPU
- `--compute-type int8` — lower precision for faster inference

### 7. Policy Inference

```bash
python -m yam_teleop.scripts.run_policy \
    --env-config configs/env.yaml \
    --checkpoint path/to/model.pt
```

Same gym.Env, same observation/action space. Replace `load_policy()` in the
script with your actual policy loader.

## Configuration

Each component has its own YAML config in `configs/`. Key parameters:

| File | What to Tune |
|------|-------------|
| `camera.yaml` | Device IDs, resolution, exposure, white balance |
| `gello.yaml` | Serial ports, joint offsets/signs (per-arm calibration) |
| `robot.yaml` | CAN channels, gripper type, gripper_max_open, home position |
| `broker.yaml` | ZMQ ports, staleness threshold |
| `env.yaml` | Home position, reset speed, gripper limits, control frequency |

## Safety

- **Gradual reset**: Arms interpolate to home position at `reset_max_delta`
  (default 0.01 rad/step) at the broker rate (~60 Hz). No sudden movements.
- **Gripper limit**: Hardware can't reach 1.0; commands are clamped at 0.85
  in the robot_node command handler.
- **Safe shutdown**: On Ctrl+C, robot_node gradually returns arms to home
  before powering off motors. A second Ctrl+C during the return stops it
  immediately at the current pose.
- **Fail-stop broker**: If the robot stream goes stale (>50 ms), the broker
  halts (stops publishing), waits for recovery, then resumes.
- **GELLO gripper-squeeze gate**: Teleop only starts after the operator
  squeezes both GELLO triggers, preventing accidental motion if the operator
  isn't ready.

## Package Structure

```
yam_teleop/
├── README.md
├── setup.py
├── configs/
│   ├── camera.yaml
│   ├── gello.yaml
│   ├── robot.yaml
│   ├── broker.yaml
│   └── env.yaml
└── yam_teleop/
    ├── __init__.py
    ├── env.py                # YAMBimanualEnv (gym.Env)
    ├── foot_pedal.py         # multiplexed evdev reader for foot pedals
    ├── video.py              # streaming ffmpeg writer (one subprocess per camera)
    ├── nodes/
    │   ├── __init__.py
    │   ├── camera_node.py
    │   ├── gello_node.py
    │   ├── robot_node.py
    │   └── sync_broker.py
    └── scripts/
        ├── __init__.py
        ├── collect_data.py
        ├── record_audio.py
        ├── record_home_pose.py
        ├── replay_episode.py
        ├── run_policy.py
        └── transcribe_audio.py
```
