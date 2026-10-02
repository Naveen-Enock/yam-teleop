# YAM Leader-Follower Teleop — Setup & Operation Guide

How to set up and run the bimanual YAM leader-follower teleop + data-collection
system on a fresh robot host.

---

## 1. Hardware

| Part | Qty | Notes |
|------|-----|-------|
| YAM follower arms | 2 | the robots being controlled; `linear_4310` grippers |
| YAM leader arms | 2 | backdriven by the operator; **teaching handles** (trigger + 2 buttons), no powered gripper |
| CAN adapters (CANable) | 4 | one per arm |
| USB cameras | 3 | `top`, `left_wrist`, `right_wrist` (optional for dev — see mock mode) |
| USB foot pedals (PCsensor FootSwitch) | 1 set | success/failure + marker (optional — keyboard fallback exists) |

The follower's `linear_4310` gripper **auto-calibrates on startup** (opens/closes
to find its range) — keep grippers clear when launching `robot_node`.

---

## 2. One-time setup

### 2a. Environment
A few dependencies build from source (`pyaudio`, and i2rt's `ruckig`), so
install a C/C++ toolchain and PortAudio first:
```bash
sudo apt install build-essential portaudio19-dev
```
This repo is a [uv](https://docs.astral.sh/uv/) workspace. From the repo root:
```bash
uv sync
```
Prefix commands with `uv run`, or `source .venv/bin/activate` once per shell.

The YAM SDK is upstream [i2rt](https://github.com/i2rt-robotics/i2rt), pinned to
a release tag under `[tool.uv.sources]` in the root `pyproject.toml`. To upgrade
i2rt, bump that tag, `uv sync`, and re-check
`yam_teleop/hardware/i2rt_compat.py` (the only code that touches i2rt
internals).

**No hardware?** Set `sim: true` in `robot.yaml` and `leader.yaml` to run the
arms as i2rt MuJoCo sims, and use `--camera-mock`. The whole node graph runs,
including the reset drive; the sim leaders have no teaching handle, so the
"squeeze both triggers" step never completes.

### 2b. Configure CAN bus names (critical)
The code matches arms by **persistent CAN interface name**, so each adapter must
come up as a fixed name:

| Arm | CAN name |
|-----|----------|
| Left follower | `can_follower_l` |
| Right follower | `can_follower_r` |
| Left leader | `can_leader_l` |
| Right leader | `can_leader_r` |

Follow i2rt's [persistent-CAN-ID guide](https://github.com/i2rt-robotics/i2rt/blob/main/docs/guides/set-persistent-can-ids.md)
to assign these names (udev rules keyed on each adapter's serial). Plug in **one
adapter at a time** while assigning. Verify all four are up:
```bash
ip -br link show type can
# can_follower_l  UP ... / can_follower_r UP ... / can_leader_l UP ... / can_leader_r UP
```
If a bus is `DOWN` after a power cycle, bring it up (e.g. `sudo ip link set <name> up type can bitrate 1000000`, or i2rt's [`scripts/reset_all_can.sh`](https://github.com/i2rt-robotics/i2rt/blob/main/scripts/reset_all_can.sh)). The names **must** match the configs below.

### 2c. Configs (`yam_teleop/configs/`)
| File | Set |
|------|-----|
| `robot.yaml` | follower CAN channels (`can_follower_l/r`), `gripper_type: linear_4310`, `gripper_max_open`, `gripper_max_force` |
| `leader.yaml` | leader CAN channels (`can_leader_l/r`), `gravity_comp_factor` (1.3; per-arm override if an arm drifts), `bilateral_kp`, `gripper_invert` |
| `camera.yaml` | camera `device_id`s, resolution/exposure (real cameras) |
| `camera_mock.yaml` | nothing — synthetic cameras for dev without USB cameras |
| `env.yaml` | `home_position` (raised start pose), reset speed, control freq |

---

## 3. Running the system

Each node runs in its own terminal, from the repo root, prefixed with `uv run`.
Launch order: **camera → follower → leader → broker → collection**.

```bash
# 1. Cameras  (real)
uv run python -m yam_teleop.nodes.camera_node --config yam_teleop/configs/camera.yaml
#    ...or, with NO USB cameras (dev): synthetic frames
uv run python -m yam_teleop.nodes.camera_node --config yam_teleop/configs/camera_mock.yaml

# 2. Followers.  ⚠️ grippers auto-calibrate on startup — keep them clear.
uv run python -m yam_teleop.nodes.robot_node --config yam_teleop/configs/robot.yaml

# 3. Leaders (teaching handles).
uv run python -m yam_teleop.nodes.yam_leader_node --config yam_teleop/configs/leader.yaml

# 4. Sync broker.
uv run python -m yam_teleop.nodes.sync_broker --config yam_teleop/configs/broker.yaml

# 5. Data collection.
uv run python -m yam_teleop.scripts.collect_yam \
    --env-config yam_teleop/configs/env.yaml --output-dir data/<task_name>
```

**Quick hardware sanity check without cameras/broker/collection** — just the two
arm nodes + a minimal teleop loop. It runs the *same* start sequence as
`collect_yam` (follower → start pose, leader waypoint → match, squeeze to start),
reading `env.yaml` for the home pose + reset waypoint:
```bash
uv run python -m yam_teleop.scripts.teleop_min      # run with robot_node + yam_leader_node
```

Shutdown order: stop the collection script first, then leader, then **follower
last** (it does a gradual safe return-to-home before motors off; a second Ctrl+C
stops it immediately).

---

## 4. Teaching-handle usage

Each leader handle has a **trigger** and **two buttons**:

- **Trigger** → the follower gripper. Squeeze = close, release = open.
- **Button 0 (marker)** → drop a sub-task boundary marker into the recording.
- **Button 1 (clutch)** → pause/resume teleop. Press once to **suspend**: the
  follower freezes and the leader goes free (move it anywhere — recording is
  paused). Press again to **resume**: the leader eases gently back to the
  follower pose (a smooth, no-jerk return — tune its speed with `resume_speed`),
  then control re-engages. Use it to recenter your arm mid-episode.

(Either handle's buttons work — they're global. If a button is mapped to the
wrong function on your handles, flip the indices in `yam_leader_node.py`.)

### Data-collection cycle (per episode)
1. **Follower reset** — arms move to the raised start pose, grippers open then close.
2. **Leader match** — the leaders drive themselves to the follower pose; then
   **squeeze both triggers** to start teleop + recording.
3. **Teleop** — move the leaders; followers track. Tap **button 0** at each
   sub-task transition. Use **button 1** to clutch out/in as needed.
4. **End** — **right foot pedal / `s`** = save SUCCESS; **left foot pedal / `f`**
   = save FAILURE (saved with a `FAILED_` prefix). `m` also drops a marker.
   `q` quits. (Foot pedals optional; keyboard keys work if none are attached.)

---

## 5. Tuning (on hardware)

In `leader.yaml`:
- **`gravity_comp_factor`** (set to `1.3`) — leader buoyancy; a scalar or 6
  per-joint values. Remove it to use i2rt's per-joint YAM default
  `[1.0, 1.1, 1.1, 1.2, 1.0, 1.0]`. 1.3 suits healthy arms; if an arm floats
  **up**, lower it for *that arm* via a per-arm override:
  ```yaml
  left:
    can_channel: can_leader_l
    gravity_comp_factor: 1.0
  ```
- **`use_coulomb_friction`** (default `false`) — i2rt's friction feedforward;
  lightens backdrive. Try it before raising `dither_amp`.
- **`bilateral_kp`** (default `0.1`) — force feedback strength (i2rt rec 0.1–0.2).
  `0` = pure passive (no feedback).
- **`kp` / `kd`** (optional) — leader PD gains for the stiff reset/resume drives.
  Unset = i2rt's YAM defaults (kp `[80,80,80,10,10,10]`).
- **`gripper_invert`** (default `true`) — if the follower gripper *opens* when you
  squeeze, set `false`.
- **`watchdog_timeout`** (default `1.0` s) — if the running collection/teleop
  script stops heartbeating (e.g. hard-killed), the leader auto-drops to limp
  gravity-comp after this long.
- **`resume_speed`** (default `0.4` rad/s) / **`resume_min_duration`** (default
  `0.5` s) — how the leader returns to the follower after a clutch-out. The
  return is a gentle smoothstep (zero jerk); raise `resume_speed` for a snappier
  return, lower it for an even softer one. `resume_min_duration` keeps tiny
  returns from being abrupt.

In `robot.yaml`:
- **`gripper_max_open`** — command-space open extent for the linear gripper.
- **`gripper_max_force`** — lower = safer on rigid objects.
- **`gripper_torque_cap`** — hard cap (Nm) on stalled-gripper torque.
- **`kp` / `kd`, `gravity_comp_factor`** (optional) — follower tracking gains and
  gravity comp. Unset = i2rt's YAM defaults. The pre-upgrade i2rt fork used
  kp 40 on joint 3 (now 10) and a flat 1.3 gravity factor — uncomment the
  values in `robot.yaml` if wrist tracking feels soft.

In `env.yaml`:
- **`leader_reset_waypoint`** — 6-joint pose the leaders pass through on reset
  (before matching the follower start pose) so the teaching handle clears the
  arm links. To re-tune: move a leader by hand to a safe clearing pose, read its
  joints off `yam_leader_node --debug`, and paste them here. Remove to skip.

---

## 6. Recorded data

```
data/<task>/<task>_YYYYMMDD_HHMMSS/        # FAILED_<...> if failure
├── episode.hdf5
│   ├── actions              (T, 14)   [left_arm(6), left_grip, right_arm(6), right_grip]
│   ├── timestamps/          camera_ns, robot_ns, gello_ns, broker_ns
│   ├── robot/               left|right joint_pos/vel/eff, gripper_pos/eff
│   ├── gello/               leader stream (joint_pos, gripper_pos) — legacy group name
│   ├── subtask_markers      (M,) step indices of marker presses
│   ├── images/              per-camera attrs (frames live in the mp4s)
│   └── attrs: num_steps, success, created_at, ...
├── top.mp4  left_wrist.mp4  right_wrist.mp4
```

The leader stream is stored under the `gello` group (and `gello_ns`) so existing
converters/labelers keep working — it's just a legacy label now.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| A CAN bus missing from `ip -br link show type can` | adapter unplugged / name not assigned / bus down — see §2b |
| Leader arm **floats upward** | `gravity_comp_factor` too high for that (worn) arm — lower it per-arm |
| Follower gripper opens when you squeeze | set `gripper_invert: false` in `leader.yaml` |
| Marker/clutch button does the wrong thing | buttons swapped on your handle — swap `io_inputs` indices in `yam_leader_node.py` |
| `collect_yam` says "no pedals found" | expected without FootSwitch; use keyboard `s`/`f`/`m`/`q` |
| Follower grippers move at startup | normal — `linear_4310` auto-calibration; keep them clear |
| Leader/follower jump at teleop start | they weren't matched — let the reset "match" finish before squeezing |
| Leader suddenly goes limp mid-use | watchdog fired — the controlling script stopped/lost its heartbeat; relaunch it |
| A node exits with `i2rt control loop stopped (motor comms lost?)` | i2rt fails fast on lost CAN comms — check that arm's power / CAN cable / `ip -br link`, then restart the node |
