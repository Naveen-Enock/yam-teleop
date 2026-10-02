# yam-teleop

A decoupled, bimanual **leader–follower teleoperation and data-collection**
system for [I2RT YAM](https://i2rt.com/) arms.

Two 6-DOF YAM follower arms are driven by a pair of leader arms, with USB
cameras and foot pedals, so an operator can teleoperate the robot and record
demonstrations for imitation learning.

The system is built as independent ZMQ microservice nodes (camera / robot /
leader / sync-broker) joined behind a single `gymnasium` environment, so the
exact same interface is used for both data collection and policy inference.

## Leader hardware

Both leader types publish the same state stream and accept the same commands,
so everything downstream (followers, broker, env, recording) is shared. Pick
one at launch:

| Leader | Node | Launch | Collection script |
|--------|------|--------|-------------------|
| **YAM teaching-handle arms** (default) — gravity-comp backdrive, bilateral force feedback, clutch + marker buttons | `yam_leader_node` | `./launch_nodes.sh` | `collect_yam` |
| **GELLO** (Dynamixel) arms — passive leaders, audio-pedal markers | `gello_node` | `./launch_nodes.sh --leader gello` | `collect_data` |

The GELLO driver is an optional extra: `uv sync --extra gello`.

## Layout

```
yam_teleop/            the teleop package (nodes, scripts, configs)
  yam_teleop/hardware/ follower wrapper, i2rt compat layer, vendored GELLO driver
```

The YAM SDK is upstream [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt)
(MIT), pinned to a release tag in `pyproject.toml` — no fork or submodule. The
GELLO Dynamixel driver is vendored from
[wuphilipp/gello_software](https://github.com/wuphilipp/gello_software) (MIT)
under `yam_teleop/yam_teleop/hardware/gello/`.

## Getting started

```bash
git clone <this-repo>
cd yam-teleop
```

Then follow [`yam_teleop/README.md`](yam_teleop/README.md) for installation and
usage, and [`yam_teleop/SETUP_GUIDE.md`](yam_teleop/SETUP_GUIDE.md) for hardware
setup.

The GELLO-only codebase as it was before the leaders were unified is tagged
`gello-legacy`.
