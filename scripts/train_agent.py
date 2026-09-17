"""CLI entry point: train + evaluate the RL agent against baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.parking_gym_env import ParkingLotEnv          # noqa: E402
from rl.baselines import evaluate_baselines              # noqa: E402
from rl.train import evaluate, train                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ParkEasy RL agent")
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--skip-train", action="store_true",
                    help="Only evaluate an existing model")
    args = ap.parse_args()

    if not args.skip_train:
        print(f"Training PPO for {args.timesteps:,} timesteps ...")
        model = train(total_timesteps=args.timesteps)
    else:
        from stable_baselines3 import PPO
        from core import config
        model = PPO.load(config.MODEL_PATH)
        print(f"Loaded existing model {config.MODEL_PATH}")

    print("\n=== Evaluation ===")
    rl = evaluate(model, episodes=args.eval_episodes)
    print("RL (PPO)        :", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rl.items()}))

    env = ParkingLotEnv()
    base = evaluate_baselines(env, episodes=args.eval_episodes)
    for name, stats in base.items():
        print(f"{name:<16}:", json.dumps(stats))

    rl_rr = rl["rejection_rate"]
    best_base = min(b["rejection_rate"] for b in base.values())
    print(f"\nRejection rate: RL={rl_rr:.2%}  vs best baseline={best_base:.2%}")


if __name__ == "__main__":
    main()
