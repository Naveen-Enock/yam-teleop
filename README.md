# yam-teleop

A decoupled, bimanual **leader–follower teleoperation and data-collection**
system for [I2RT YAM](https://i2rt.com/) arms.

Two 6-DOF YAM follower arms are driven by leader arms (GELLO Dynamixel leaders
on `main`; YAM teaching-handle leaders on the `feat/yam-leader-follower`
branch), with USB cameras and foot pedals, so an operator can teleoperate the
robot and record demonstrations for imitation learning.

The system is built as independent ZMQ microservice nodes (camera / robot /
leader / sync-broker) joined behind a single `gymnasium` environment, so the
exact same interface is used for both data collection and policy inference.

## Layout

```
yam_teleop/            the teleop package (nodes, scripts, configs)
third_party/
  gello_software/      vendored GELLO driver fork (submodule)
  i2rt/                vendored YAM SDK fork (submodule)
```

`third_party/gello_software` and `third_party/i2rt` are git submodules pinned to
lightly-patched forks of [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)
and [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) (both MIT). Each
fork keeps upstream history intact and adds only the small patches this project
needs on top.

## Getting started

```bash
git clone --recurse-submodules <this-repo>
cd yam-teleop
```

Then follow [`yam_teleop/README.md`](yam_teleop/README.md) for installation and
usage.

## Branches

- **`main`** — GELLO-leader teleop + data collection + local audio-narration
  capture and transcription.
- **`feat/yam-leader-follower`** — bilateral YAM-leader rearchitecture
  (teaching-handle leaders, gripper force/torque shaping, uv packaging). See
  [`yam_teleop/SETUP_GUIDE.md`](yam_teleop/SETUP_GUIDE.md) on that branch.
