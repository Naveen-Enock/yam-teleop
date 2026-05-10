"""Policy inference script.

Same gym env as data collection, but actions come from a policy network
instead of GELLO leader arms.

Usage:
    python -m yam_teleop.scripts.run_policy --env-config configs/env.yaml --checkpoint path/to/model.pt
"""

import argparse
import time

import numpy as np

from yam_teleop.env import YAMBimanualEnv


def load_policy(checkpoint_path: str):
    """Load a policy from checkpoint. Replace with your actual policy loader."""
    raise NotImplementedError(
        "Replace this with your policy loading code. "
        "The policy should accept an observation dict and return a 14-dim action."
    )


def main():
    parser = argparse.ArgumentParser(description="Run policy inference")
    parser.add_argument("--env-config", required=True, help="Path to env.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to policy checkpoint")
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    env = YAMBimanualEnv(args.env_config)
    policy = load_policy(args.checkpoint)

    try:
        obs, _ = env.reset()

        for step in range(args.max_steps):
            action = policy(obs)
            action = np.asarray(action, dtype=np.float64).reshape(14)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 60 == 0:
                print(f"Step {step}")

            if terminated or truncated:
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
