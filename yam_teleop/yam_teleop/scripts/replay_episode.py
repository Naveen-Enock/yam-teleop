"""Replay a recorded episode on the YAM robot arms.

Loads actions from an HDF5 episode file and replays them through the env
at the broker's 60Hz rate. The robot physically executes the recorded
GELLO commands.

Usage:
    python -m yam_teleop.scripts.replay_episode --env-config configs/env.yaml --episode data/task/episode_0000.hdf5
"""

import argparse
import signal
import sys
import time

import cv2
import h5py
import numpy as np
import zmq

from yam_teleop.env import YAMBimanualEnv


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded episode")
    parser.add_argument("--env-config", required=True,
                        help="Path to env.yaml")
    parser.add_argument("--episode", required=True,
                        help="Path to episode HDF5 file")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live camera preview")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip reset (assume arms are already at home)")
    parser.add_argument("--gello-cmd-port", type=int, default=5006,
                        help="ZMQ port for GELLO commands (default: 5006)")
    args = parser.parse_args()

    show_display = not args.no_display

    # Load actions from episode
    with h5py.File(args.episode, "r") as f:
        actions = f["actions"][:]
        num_steps = f.attrs["num_steps"]
    print(f"Loaded episode: {args.episode}")
    print(f"  {num_steps} steps ({num_steps / 60:.1f}s at 60Hz)")
    print(f"  Action shape: {actions.shape}")

    env = YAMBimanualEnv(args.env_config)

    # GELLO command socket (for safety: disable torque on exit)
    gello_cmd_ctx = zmq.Context()
    gello_cmd = gello_cmd_ctx.socket(zmq.PUSH)
    gello_cmd.setsockopt(zmq.LINGER, 0)
    gello_cmd.connect(f"tcp://127.0.0.1:{args.gello_cmd_port}")
    time.sleep(0.5)

    # Graceful shutdown
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        # Reset to home
        if not args.no_reset:
            print("Resetting to home position...")
            obs, info = env.reset()
            if info.get("reset_failed"):
                print("Reset failed. Fix arm positions and try again.")
                return
            print(">> Reset complete. Starting replay in 2s...")
            time.sleep(2.0)
        else:
            obs = env._get_obs()

        print(f">> Replaying {num_steps} steps...")
        t_start = time.time()

        for step in range(num_steps):
            if not running:
                print(f"\n>> Interrupted at step {step}")
                break

            action = actions[step].astype(np.float64)
            obs, _, _, _, _ = env.step(action)

            if step < 5 or (step + 1) % 60 == 0:
                elapsed = time.time() - t_start
                print(f"  Step {step + 1}/{num_steps} "
                      f"({elapsed:.1f}s elapsed)")

            # Live camera preview
            if show_display:
                frames = []
                for cam_name in sorted(obs["images"].keys()):
                    img = obs["images"][cam_name]
                    small = cv2.resize(img, (640, 360))
                    cv2.putText(small, cam_name, (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)
                    frames.append(small)
                if frames:
                    canvas = np.hstack(frames)
                    cv2.imshow("Replay", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print(f"\n>> Quit at step {step + 1}")
                        break

        elapsed = time.time() - t_start
        print(f">> Replay finished: {num_steps} steps in {elapsed:.1f}s "
              f"({num_steps / elapsed:.1f} Hz)")

    except KeyboardInterrupt:
        print("\n>> Interrupted")
    finally:
        # Ensure GELLO torque is off (prevents 200Hz loop slowdown if left on)
        try:
            gello_cmd.send_json({"command": "disable_torque"})
        except Exception:
            pass
        if show_display:
            cv2.destroyAllWindows()
        gello_cmd.close()
        gello_cmd_ctx.term()
        env.close()


if __name__ == "__main__":
    main()
