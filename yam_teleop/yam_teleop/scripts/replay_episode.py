"""Replay a recorded episode on the follower arms.

Every saved episode (collect_yam / collect_data) stores the commanded joint
targets for both arms + grippers at 60 Hz in its HDF5 ``actions`` dataset.
This script sends them back to robot_node at the same 60 Hz, so the followers
retrace the recorded motion. It is open-loop: objects must be where they were
when the episode was recorded.

Needs the same nodes as collection (cameras, followers, broker; the leaders
can stay up and are left free) but NOT the collect script — both send the
follower commands, so stop pane [5] first.

Sequence: reset to home (as at the start of every recorded episode) -> ease
to the episode's first pose -> wait for Enter -> replay. Press q in the
preview window or Ctrl+C to stop; the arms hold where they are.

Usage:
    uv run python -m yam_teleop.scripts.replay_episode \\
        --episode data/<task>/episode_0000.hdf5
"""

import argparse
import errno
import signal
import time

import cv2
import h5py
import numpy as np
import zmq

from yam_teleop.env import YAMBimanualEnv

FPS = 60.0


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode")
    parser.add_argument("--episode", required=True,
                        help="Path to episode HDF5 file")
    parser.add_argument("--env-config", default="yam_teleop/configs/env.yaml",
                        help="Path to env.yaml (default: %(default)s)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start this many seconds into the episode")
    parser.add_argument("--end", type=float, default=None,
                        help="Stop this many seconds into the episode")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live camera preview")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip the reset to home (still eases to the "
                             "first pose)")
    parser.add_argument("--yes", action="store_true",
                        help="Don't wait for Enter before replaying")
    parser.add_argument("--leader-cmd-port", "--gello-cmd-port", type=int,
                        default=5006,
                        help="Leader node command port, used to leave the "
                             "leaders free on exit (default: 5006)")
    args = parser.parse_args()

    show_display = not args.no_display

    with h5py.File(args.episode, "r") as f:
        actions = f["actions"][:].astype(np.float64)
        success = f.attrs.get("success", None)
        markers = [int(m) for m in f["subtask_markers"][:]] if "subtask_markers" in f else []
    n = len(actions)
    first = min(max(int(round(args.start * FPS)), 0), max(n - 1, 0))
    last = n if args.end is None else min(n, max(int(round(args.end * FPS)), first + 1))
    print(f"Loaded {args.episode}")
    print(f"  {n} steps ({n / FPS:.1f}s at {FPS:.0f} Hz), action shape "
          f"{actions.shape}, success={success}, markers={markers}")
    if actions.ndim != 2 or actions.shape[1] != 14 or n == 0:
        raise SystemExit("Expected a non-empty (N, 14) actions dataset.")
    if first or last != n:
        print(f"  Replaying steps {first}..{last - 1} "
              f"({first / FPS:.1f}s to {last / FPS:.1f}s)")

    try:
        env = YAMBimanualEnv(args.env_config)
    except zmq.ZMQError as e:
        if e.errno == errno.EADDRINUSE:
            raise SystemExit(
                "Robot command port is already in use — the collect script "
                "(pane [5]) is probably still running. Stop it (q or Ctrl+C) "
                "and run the replay again.")
        raise

    leader_ctx = zmq.Context()
    leader_cmd = leader_ctx.socket(zmq.PUSH)
    leader_cmd.setsockopt(zmq.LINGER, 0)
    leader_cmd.connect(f"tcp://127.0.0.1:{args.leader_cmd_port}")

    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, shutdown)

    try:
        if not args.no_reset:
            print("Resetting to home...")
            env.reset()
        print("Easing to the episode's first pose...")
        env.move_to(actions[first], "Moving to start pose...")

        if not args.yes:
            input(">> Ready. Press Enter to replay (Ctrl+C to cancel)... ")

        steps = last - first
        print(f">> Replaying {steps} steps ({steps / FPS:.1f}s)...")
        t_start = time.time()
        marker_set = set(markers)
        step = first
        for step in range(first, last):
            if not running:
                print(f"\n>> Stopped at step {step}")
                break

            obs, _, _, _, _ = env.step(actions[step])

            if step in marker_set:
                print(f"  marker at step {step} ({step / FPS:.1f}s)")
            if (step - first + 1) % 60 == 0:
                print(f"  {step / FPS:5.1f}s / {last / FPS:.1f}s  "
                      f"(wall {time.time() - t_start:.1f}s)")

            if show_display:
                frames = []
                for cam_name in sorted(obs["images"].keys()):
                    small = cv2.resize(obs["images"][cam_name], (640, 360))
                    cv2.putText(small, cam_name, (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    frames.append(small)
                if frames:
                    cv2.imshow("Replay", np.hstack(frames))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print(f"\n>> Quit at step {step}")
                        break
        else:
            elapsed = time.time() - t_start
            print(f">> Replay finished: {steps} steps in {elapsed:.1f}s "
                  f"({steps / max(elapsed, 1e-6):.1f} Hz). Arms are holding "
                  f"the last pose.")

    except KeyboardInterrupt:
        print("\n>> Interrupted — arms are holding where they are.")
    finally:
        # Leave the leaders free (no torque) whatever state they were in.
        try:
            leader_cmd.send_json({"command": "disable_torque"})
        except Exception:
            pass
        if show_display:
            cv2.destroyAllWindows()
        leader_cmd.close()
        leader_ctx.term()
        env.close()


if __name__ == "__main__":
    main()
