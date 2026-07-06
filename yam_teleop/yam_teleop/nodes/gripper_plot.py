"""Live plot of the follower's gripper position, velocity and torque.

Enabled by ``robot_node --plot-gripper``. matplotlib runs in a SEPARATE
PROCESS (not a thread) so the 200Hz control loop that drives the physical
arms never blocks on GUI rendering and never contends for the GIL. The hot
loop feeds samples through a small ``multiprocessing.Queue`` with a
non-blocking put that silently drops when the plotter can't keep up — so
pushing a sample costs microseconds and can never stall teleop.

Layout: position (top-left) and velocity (top-right), with torque spanning the
full bottom row. Each panel shows both the left and right grippers. Torque is
the DM4310's measured joint effort in Nm; the motor exposes no current channel,
so current is deliberately not plotted (see the --plot-gripper discussion).

We use the ``spawn`` start method deliberately: ``robot_node`` already has a
command-receiver thread running by the time the plotter starts, and forking a
multithreaded process can deadlock inside the child's GUI/toolkit init. Spawn
gives the plotter a clean interpreter. The plot target lives in this module
(not in ``robot_node.__main__``) so spawn re-imports only this side-effect-free
module and never re-runs the node's ``main()``.

Needs an interactive matplotlib backend (e.g. Qt via PyQt5) and a display; if
neither is available the child says so and idles instead of erroring.
"""

from __future__ import annotations

import multiprocessing as mp

# matplotlib backends that can't open a window — if we resolve to one of these
# there's no point drawing (headless / no GUI toolkit installed).
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "pgf", "cairo"}


def _fit_y_axis(ax, series) -> None:
    """Scale ``ax``'s y-range to the signal ``series`` (iterables) only, with a
    small margin. Reference overlays are excluded, so they never widen the axis
    — they're just shown or clipped depending on where the data lands."""
    vals = [v for s in series for v in s]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:  # flat signal -> give it a little room
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)


def _plot_process(queue, window_sec: float, sample_hz: float,
                  speed_limit=None, hold_band=None) -> None:
    """Child-process entry point: drain the queue and redraw the subplots."""
    # Imported here, inside the child, so the parent (the 200Hz loop) never
    # pays the matplotlib import cost and no GUI state crosses the process
    # boundary.
    import collections
    import time
    from queue import Empty

    import matplotlib
    import matplotlib.pyplot as plt

    # Bail early (and clearly) if there's no GUI backend to draw into — e.g.
    # headless, or no Qt/Tk toolkit installed in this environment.
    backend = matplotlib.get_backend()
    if backend.lower() in _NON_INTERACTIVE_BACKENDS:
        print(f"[gripper_plot] non-interactive matplotlib backend "
              f"({backend!r}); no window. Install an interactive backend "
              f"(e.g. `uv pip install pyqt5`) and run with a display to see "
              f"the live gripper plot.")
        # Keep draining so the parent's non-blocking puts never wedge on a
        # full queue; exit when it sends the shutdown sentinel.
        while True:
            try:
                if queue.get(timeout=1.0)[0] is None:
                    return
            except Empty:
                pass
            except Exception:
                return

    maxlen = int(window_sec * sample_hz) + 1
    t_buf = collections.deque(maxlen=maxlen)
    l_pos = collections.deque(maxlen=maxlen)
    l_vel = collections.deque(maxlen=maxlen)
    l_tau = collections.deque(maxlen=maxlen)
    r_pos = collections.deque(maxlen=maxlen)
    r_vel = collections.deque(maxlen=maxlen)
    r_tau = collections.deque(maxlen=maxlen)

    plt.ion()
    # position | velocity on top, torque spanning the full bottom row.
    fig, axd = plt.subplot_mosaic(
        [["pos", "vel"],
         ["tau", "tau"]],
        figsize=(12, 7),
    )
    ax_pos, ax_vel, ax_tau = axd["pos"], axd["vel"], axd["tau"]
    fig.suptitle("Follower gripper — live", fontsize=13)

    (l_pos_line,) = ax_pos.plot([], [], label="left", lw=1.2)
    (r_pos_line,) = ax_pos.plot([], [], label="right", lw=1.2)
    (l_vel_line,) = ax_vel.plot([], [], label="left", lw=1.2)
    (r_vel_line,) = ax_vel.plot([], [], label="right", lw=1.2)
    (l_tau_line,) = ax_tau.plot([], [], label="left", lw=1.2)
    (r_tau_line,) = ax_tau.plot([], [], label="right", lw=1.2)

    ax_pos.set_title("Gripper position")
    ax_pos.set_xlabel("time (s)")
    # i2rt normalizes the gripper joint to [0,1] (0=closed, 1=open) via the
    # auto-calibrated closed/open limits; it is NOT radians. Velocity is the
    # follower's finite-difference of that fraction -> stroke fraction / s.
    ax_pos.set_ylabel("position (0 = closed, 1 = open)")

    ax_vel.set_title("Gripper velocity")
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_ylabel("velocity (stroke fraction / s)")

    # Torque is the DM4310's measured joint effort (joint_efforts[6]) in Nm;
    # the motor reports no current, so this is the real load signal.
    ax_tau.set_title("Gripper torque")
    ax_tau.set_xlabel("time (s)")
    ax_tau.set_ylabel("torque (Nm)")

    # Reference overlays. These do NOT drive the y-scale (see the redraw loop):
    # the axes autoscale to the signal, and each reference is only shown when it
    # already falls within that data-driven range. The slew cap is on the
    # *commanded* gripper position, so measured velocity may briefly exceed it
    # (finite-diff noise) — it's a reference, not a hard bound on this trace.
    vel_ref_hi = vel_ref_lo = None
    tau_band_hi = tau_band_lo = None
    if speed_limit and speed_limit > 0:
        vel_ref_hi = ax_vel.axhline(speed_limit, color="crimson", lw=1.0,
                                    ls="--", alpha=0.7, label="cmd slew limit (±)")
        vel_ref_lo = ax_vel.axhline(-speed_limit, color="crimson", lw=1.0,
                                    ls="--", alpha=0.7)
    if hold_band:
        blo, bhi = hold_band
        tau_band_hi = ax_tau.axhspan(blo, bhi, color="orange", alpha=0.15,
                                     label="force-limit hold (±)")
        tau_band_lo = ax_tau.axhspan(-bhi, -blo, color="orange", alpha=0.15)

    for ax in (ax_pos, ax_vel, ax_tau):
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    try:
        fig.show()
    except Exception as e:  # display vanished mid-setup -> fail gracefully
        print(f"[gripper_plot] could not open window "
              f"(backend={backend!r}): {e}")
        return

    # Manual interactive loop (not FuncAnimation): we fully own shutdown, so
    # closing the window or receiving the sentinel exits cleanly with no
    # dangling-timer traceback. Redraw is capped to ~30 FPS independent of the
    # incoming sample rate.
    last_draw = 0.0
    running = True
    while running:
        try:
            sample = queue.get(timeout=0.2)
        except Empty:
            if not plt.fignum_exists(fig.number):  # user closed the window
                break
            fig.canvas.flush_events()
            continue
        except Exception:
            break

        # Drain everything queued since the last redraw.
        while True:
            if sample[0] is None:  # sentinel -> parent asked us to close
                running = False
                break
            t, lp, lv, lt, rp, rv, rt = sample
            t_buf.append(t)
            l_pos.append(lp)
            l_vel.append(lv)
            l_tau.append(lt)
            r_pos.append(rp)
            r_vel.append(rv)
            r_tau.append(rt)
            try:
                sample = queue.get_nowait()
            except Empty:
                break

        if not running or not plt.fignum_exists(fig.number):
            break

        now = time.monotonic()
        if t_buf and now - last_draw >= 0.033:
            t0 = t_buf[0]
            x = [ti - t0 for ti in t_buf]
            l_pos_line.set_data(x, l_pos)
            r_pos_line.set_data(x, r_pos)
            l_vel_line.set_data(x, l_vel)
            r_vel_line.set_data(x, r_vel)
            l_tau_line.set_data(x, l_tau)
            r_tau_line.set_data(x, r_tau)

            # Rolling x-window shared by all panels.
            if len(x) >= 2:
                for ax in (ax_pos, ax_vel, ax_tau):
                    ax.set_xlim(x[0], x[-1])
            # Autoscale each panel to its own signal (references excluded).
            _fit_y_axis(ax_pos, (l_pos, r_pos))
            _fit_y_axis(ax_vel, (l_vel, r_vel))
            _fit_y_axis(ax_tau, (l_tau, r_tau))
            # Show each reference only if it now lies within the visible range.
            if vel_ref_hi is not None:
                lo, hi = ax_vel.get_ylim()
                vel_ref_hi.set_visible(lo <= speed_limit <= hi)
                vel_ref_lo.set_visible(lo <= -speed_limit <= hi)
            if tau_band_hi is not None:
                lo, hi = ax_tau.get_ylim()
                blo, bhi = hold_band
                tau_band_hi.set_visible(bhi >= lo and blo <= hi)   # +band overlaps
                tau_band_lo.set_visible(-blo >= lo and -bhi <= hi)  # -band overlaps
            fig.canvas.draw_idle()
            last_draw = now
        fig.canvas.flush_events()

    plt.close(fig)


class GripperPlotter:
    """Handle to a separate-process live gripper plot.

    Feed it with :meth:`push` from the control loop; samples are dropped
    silently when the plotter is behind, so the caller never blocks. Call
    :meth:`close` to tear the window and process down.
    """

    def __init__(self, window_sec: float = 20.0, sample_hz: float = 50.0,
                 queue_size: int = 512, speed_limit=None, hold_band=None):
        """speed_limit: gripper cmd slew cap (stroke-fraction/s) drawn on the
        velocity panel. hold_band: (lo, hi) Nm force-limit band drawn on the
        torque panel. Either may be None to omit that reference."""
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=queue_size)
        self._proc = ctx.Process(
            target=_plot_process,
            args=(self._queue, window_sec, sample_hz, speed_limit, hold_band),
            daemon=True,
        )
        self._proc.start()

    def push(self, t: float,
             l_pos: float, l_vel: float, l_tau: float,
             r_pos: float, r_vel: float, r_tau: float) -> None:
        """Enqueue one sample; drop it (never block) if the queue is full."""
        try:
            self._queue.put_nowait((t, l_pos, l_vel, l_tau, r_pos, r_vel, r_tau))
        except Exception:
            pass  # queue full or closed -> drop, keep the hot loop moving

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def close(self) -> None:
        """Ask the window to close, then reap the process."""
        try:
            self._queue.put_nowait((None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
