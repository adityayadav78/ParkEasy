"""PPO training + evaluation for the parking allocation agent."""
from __future__ import annotations

import sys

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from core import config
from envs.parking_gym_env import ParkingLotEnv


def make_env(rank: int = 0, seed: int | None = None):
    def _init():
        env = ParkingLotEnv(seed=None if seed is None else seed + rank)
        return Monitor(env)
    return _init


def train(total_timesteps: int | None = None, model_path=None) -> MaskablePPO:
    total_timesteps = total_timesteps or config.RL["total_timesteps"]
    n_envs = config.RL["n_envs"]
    # SubprocVecEnv needs a __main__ guard; on Windows (spawn) or inside a
    # running server it's safer and barely slower to train in-process.
    if n_envs > 1 and sys.platform != "win32":
        try:
            venv = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        except Exception:
            venv = DummyVecEnv([make_env(i) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([make_env(i) for i in range(max(n_envs, 1))])

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        learning_rate=config.RL["learning_rate"],
        n_steps=1024,
        batch_size=512,
        gamma=config.RL["gamma"],
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=0,
        seed=7,
    )
    eval_env = DummyVecEnv([make_env(0, seed=123)])
    eval_cb = EvalCallback(eval_env, best_model_save_path=str(config.MODELS_DIR),
                           eval_freq=10_000, n_eval_episodes=3, deterministic=True)
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    model.save(config.MODEL_PATH.with_suffix(""))
    venv.close()
    return model


def evaluate(model, episodes: int = 5) -> dict:
    env = ParkingLotEnv(seed=None)
    totals = {"arrivals": 0, "accepted": 0, "rejected": 0, "revenue": 0.0, "wrong_zone": 0}
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=env.action_masks())
            obs, reward, term, trunc, info = env.step(int(action))
            done = term or trunc
        for k in totals:
            totals[k] += info["stats"][k]
    for k in totals:
        totals[k] = totals[k] / episodes
    totals["rejection_rate"] = totals["rejected"] / max(totals["arrivals"], 1)
    return totals
