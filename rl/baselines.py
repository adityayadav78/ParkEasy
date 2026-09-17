"""Greedy allocation baselines to compare against the RL agent."""
from __future__ import annotations

import numpy as np

from envs.parking_gym_env import ParkingLotEnv


def run_baseline(env: ParkingLotEnv, policy: str = "first_fit") -> dict:
    """Replay one episode, choosing actions with a fixed heuristic policy."""
    obs, info = env.reset()
    done = False
    while not done:
        valid = info["valid_actions"]
        if policy == "first_fit":
            idxs = np.where(valid[:-1])[0]
            action = int(idxs[0]) if len(idxs) else env.n_slots
        elif policy == "nearest_fit":
            # prefer matching-type zones, otherwise any allowed zone
            veh = info["vehicle"]
            preferred = {"ev": "E", "disabled": "D", "bike": "M"}.get(veh)
            order: list[int] = []
            if preferred:
                order += list(np.where(env.zone_of == preferred)[0])
            order += list(np.where(np.isin(env.zone_of, ["A", "B"]))[0])
            order += list(np.where(env.zone_of == "P")[0])
            order += [env.n_slots]  # reject last resort
            action = next((a for a in order if valid[a]), env.n_slots)
        else:
            raise ValueError(policy)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
    return info["stats"]


def evaluate_baselines(env: ParkingLotEnv, episodes: int = 3) -> dict:
    results = {}
    for pol in ("first_fit", "nearest_fit"):
        totals = {"arrivals": 0, "accepted": 0, "rejected": 0, "revenue": 0.0, "wrong_zone": 0}
        for ep in range(episodes):
            s = run_baseline(env, pol)
            for k in totals:
                totals[k] += s[k]
        for k in totals:
            totals[k] = totals[k] / episodes
        rej_rate = totals["rejected"] / max(totals["arrivals"], 1)
        results[pol] = {**{k: round(v, 1) for k, v in totals.items()},
                        "rejection_rate": round(rej_rate, 4)}
    return results
