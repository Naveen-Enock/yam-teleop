# yam_teleop (YAM leader-follower)

Bimanual **YAM leader → YAM follower** teleoperation and data collection.

Two YAM follower arms are backdriven by two YAM leader arms fitted with
teaching handles (trigger + buttons). The system runs as independent ZMQ
nodes joined behind a single `gymnasium` environment, used identically for
data collection and policy inference.

This repo is a [uv](https://docs.astral.sh/uv/) workspace on Python 3.12.
From the repo root:

```bash
uv sync                      # core teleop + data collection
uv sync --extra transcribe   # + faster-whisper transcription (GPU, optional)
```

Then launch the stack:

```bash
./launch_nodes.sh                # real USB cameras
./launch_nodes.sh --camera-mock  # synthetic frames (no USB cameras)
```

**See [`SETUP_GUIDE.md`](SETUP_GUIDE.md)** for full hardware setup, CAN bus
configuration, and operation. The vendored `i2rt` YAM SDK lives under
`../third_party/i2rt` (git submodule).

## Nodes

| Node | Role |
|------|------|
| `camera_node` | 3 USB cameras (or `--camera-mock` synthetic frames) |
| `robot_node` | 2 YAM followers via `i2rt`, gripper force/torque limiting |
| `yam_leader_node` | 2 YAM leaders (teaching handles), clutch + watchdog |
| `sync_broker` | interpolates follower state onto camera timestamps |

Data collection: `python -m yam_teleop.scripts.collect_yam`. The gym interface
is `YAMBimanualEnv` (`yam_teleop/env.py`); policy inference uses
`python -m yam_teleop.scripts.run_policy`.
