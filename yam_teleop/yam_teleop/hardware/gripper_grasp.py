"""Force-limited grasping for the follower gripper.

Replaces i2rt's ``GripperForceLimiter`` inside the i2rt control loop (~1 kHz,
fresh motor state every tick). It has the same ``update(gripper_state) ->
raw position command`` interface, so i2rt calls it in place of its own limiter.

The gripper motor runs on-board PD (MIT mode), so in normalized units
(0 = closed, 1 = open) the torque it squeezes with is

    squeeze = K * (pos - cmd) + D * vel        (closing-positive, Nm)

with K = kp * stroke and D = kd * stroke. The only way to bound the squeeze is
to keep the command close to the measured position. But moving takes torque
too: at 1 stroke/s the damping term alone eats ~3.3 Nm on the linear_4310, so a
fixed tight clamp would throttle free closing to a crawl. Hence two states:

FREE    Command = operator target, but never more than
            (cap + D * min(target_speed, gripper_speed)) / K
        past the gripper (closing speeds only). While the gripper keeps up
        with the operator's slew-limited target, that is exactly enough lead
        to cancel the damping, so free closing runs at full speed. When the
        gripper is stopped by an object its speed drops to zero and the
        allowance collapses to cap / K, so the squeeze is bounded by `cap`
        within a few ms of contact. gripper_speed is measured from encoder
        position over a 5 ms window (clean, unlike the motor's velocity
        reading) and can never exceed the commanded target speed, so it can't
        pump the gripper faster than the operator asked.
GRASP   Entered when the gripper has stalled (|speed| < stall_speed) while
        squeezing > detect_torque and the operator still asks for more, held
        for detect_time. The command is then kept `hold / K` past the measured
        position: the gripper presses with the configured grip force and no
        harder, however hard the trigger is squeezed, and stays put. The
        motor's own damping is untouched, so this is stable.
        Leaves GRASP when the operator opens past the hold point, or when the
        gripper travels `slip_travel` further closed (object slipped out or
        removed), so it can close freely again.

The state is latched with hysteresis, so it can't flicker between modes, and
the motor's noisy velocity reading is never used — an earlier attempt that fed
it into the command made the gripper chatter.
"""

from __future__ import annotations

import collections
import time
from typing import Callable

# Window for the encoder-position speed estimate (s).
_SPEED_WINDOW_S = 0.005

# i2rt adds this to its force->torque map to overcome friction; we match it.
FRICTION_COMP_NM = 0.3


def _lpf(prev: float, x: float, dt: float, tau: float) -> float:
    if dt <= 0:
        return prev
    return prev + min(1.0, dt / tau) * (x - prev)


class GripperGraspController:
    """Latching grasp controller. See module docstring."""

    def __init__(
        self,
        *,
        kp: float,
        kd: float,
        raw_closed: float,
        raw_open: float,
        hold_torque_fn: Callable[[float], float],
        torque_cap: float = 0.0,
        detect_torque: float | None = None,
        stall_speed: float = 0.05,
        detect_time: float = 0.005,
        slip_travel: float = 0.04,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        kp, kd: gripper motor PD gains (Nm/rad, Nm*s/rad).
        raw_closed, raw_open: motor positions (rad) at normalized 0 and 1.
        hold_torque_fn(raw_pos): motor torque (Nm) for the configured grip
            force at that motor position, friction comp included.
        torque_cap: ceiling (Nm) on the stalled squeeze before and during a
            grasp. <= 0 disables the cap (only the grasp hold limits force).
        detect_torque: squeeze (Nm) that counts as contact. Default:
            max(2 * friction, 1.2 * nominal hold torque), kept under the cap.
        stall_speed: |gripper speed| (stroke-fraction/s) counted as stalled.
        detect_time: how long contact must persist before latching (s).
        slip_travel: closing travel (stroke-fraction) after latching that
            means the object is gone.
        """
        self._lo = float(raw_closed)
        self._range = float(raw_open) - float(raw_closed)
        stroke = abs(self._range)
        if stroke <= 0 or kp <= 0:
            raise ValueError("gripper needs a non-zero stroke and kp")
        self._K = kp * stroke
        self._D = kd * stroke
        self._hold_torque_fn = hold_torque_fn
        self._cap = float(torque_cap) if torque_cap and torque_cap > 0 else None
        nominal_hold = self._hold_at_norm(0.5)
        if detect_torque is None:
            detect_torque = max(2 * FRICTION_COMP_NM, 1.2 * nominal_hold)
            if self._cap is not None:
                detect_torque = min(detect_torque, 0.9 * self._cap)
        self._detect = float(detect_torque)
        self._stall_speed = float(stall_speed)
        self._detect_time = float(detect_time)
        self._slip = float(slip_travel)
        self._clock = clock

        self._last_cmd: float | None = None
        self._last_target: float | None = None
        self._last_t: float | None = None
        self._target_vel = 0.0       # filtered, normalized/s (+ = opening)
        self._pos_hist: collections.deque = collections.deque(maxlen=64)
        self._pressing_since: float | None = None
        self._latch_pos = 0.0
        # Telemetry (read from other threads; plain attribute writes).
        self.grasping = False
        self.squeeze_nm = 0.0
        self.hold_nm = nominal_hold
        self.grasp_count = 0

    # -- unit helpers -----------------------------------------------------
    def _to_raw(self, n: float) -> float:
        return self._lo + n * self._range

    def _hold_at_norm(self, n: float) -> float:
        hold = abs(float(self._hold_torque_fn(self._to_raw(n))))
        return min(hold, self._cap) if self._cap is not None else hold

    def _speed(self, now: float, p: float) -> float:
        """Gripper velocity (normalized/s, + = opening) from encoder position."""
        h = self._pos_hist
        h.append((now, p))
        while len(h) > 2 and now - h[1][0] >= _SPEED_WINDOW_S:
            h.popleft()
        t0, p0 = h[0]
        return (p - p0) / (now - t0) if now - t0 > 1e-4 else 0.0

    # -- i2rt limiter interface -------------------------------------------
    def update(self, gripper_state: dict) -> float:
        """Return the raw gripper position command for this control tick."""
        p = float(gripper_state["current_normalized_qpos"])
        target = float(gripper_state["target_normalized_qpos"])
        now = self._clock()
        if self._last_t is None:
            dt = 0.0
            self._last_cmd = target
            self._last_target = target
        else:
            dt = min(max(now - self._last_t, 0.0), 0.02)
        self._last_t = now

        if dt > 0:
            self._target_vel = _lpf(self._target_vel,
                                    (target - self._last_target) / dt, dt, 0.01)
        self._last_target = target
        v = self._speed(now, p)
        squeeze = self._K * (p - self._last_cmd) + self._D * v
        self.squeeze_nm = _lpf(self.squeeze_nm, squeeze, dt, 0.003)

        hold = self._hold_at_norm(p)
        self.hold_nm = hold
        m_hold = hold / self._K

        if self.grasping:
            released = target >= p - 0.5 * m_hold
            slipped = self._latch_pos - p > self._slip
            if released or slipped:
                self.grasping = False
                self._pressing_since = None
        else:
            pressing = (self.squeeze_nm > self._detect
                        and abs(v) < self._stall_speed
                        and p - target > m_hold)  # operator wants it tighter
            if not pressing:
                self._pressing_since = None
            elif self._pressing_since is None:
                self._pressing_since = now
            elif now - self._pressing_since >= self._detect_time:
                self.grasping = True
                self.grasp_count += 1
                self._latch_pos = p

        cmd = target
        if self.grasping:
            cmd = max(target, p - m_hold)
        elif self._cap is not None:
            closing_speed = min(max(0.0, -self._target_vel), max(0.0, -v))
            lead = (self._cap + self._D * closing_speed) / self._K
            cmd = max(target, p - lead)
        # Opening is never limited; keep within the normalized range.
        cmd = min(max(cmd, 0.0), 1.0)
        self._last_cmd = cmd
        return self._to_raw(cmd)
