# yam_teleop

Bimanual **leader → YAM follower** teleoperation and data collection.

Two YAM follower arms are driven by either two YAM leader arms fitted with
teaching handles (trigger + buttons; the default) or two GELLO Dynamixel
leader arms. The system runs as independent ZMQ nodes joined behind a single
`gymnasium` environment, used identically for data collection and policy
inference.

This repo is a [uv](https://docs.astral.sh/uv/) workspace on Python 3.12.
From the repo root:

```bash
uv sync                      # core teleop + data collection
uv sync --extra transcribe   # + faster-whisper transcription (GPU, optional)
uv sync --extra gello        # + GELLO leader driver (GELLO rigs only)
```

Then launch the stack:

```bash
./launch_nodes.sh                  # YAM leaders, real USB cameras
./launch_nodes.sh --leader gello   # GELLO leaders
./launch_nodes.sh --camera-mock    # synthetic frames (no USB cameras)
```

**See [`SETUP_GUIDE.md`](SETUP_GUIDE.md)** for full hardware setup, CAN bus
configuration, and operation. The YAM SDK is upstream
[i2rt](https://github.com/i2rt-robotics/i2rt), pinned to a release tag in the
root `pyproject.toml`; the few places this package goes beyond i2rt's public
API are collected in `yam_teleop/hardware/i2rt_compat.py`.

## Nodes

| Node | Role |
|------|------|
| `camera_node` | 3 USB cameras (or `--camera-mock` synthetic frames) |
| `robot_node` | 2 YAM followers via `i2rt`, gripper force/torque limiting |
| `yam_leader_node` | 2 YAM leaders (teaching handles), clutch + watchdog |
| `gello_node` | 2 GELLO leaders (Dynamixel) — alternative to `yam_leader_node` |
| `sync_broker` | interpolates follower state onto camera timestamps |

Data collection: `python -m yam_teleop.scripts.collect_yam` (YAM leaders) or
`python -m yam_teleop.scripts.collect_data` (GELLO leaders). The gym interface
is `YAMBimanualEnv` (`yam_teleop/env.py`); policy inference uses
`python -m yam_teleop.scripts.run_policy`.
