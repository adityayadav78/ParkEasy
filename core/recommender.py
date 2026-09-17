"""Slot recommendation: trained RL policy first, heuristic fallback otherwise."""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np

from core import config
from envs.parking_gym_env import ParkingLotEnv, STATUS_IDX, VEHICLE_IDX


class SlotRecommender:
    """Wraps the SB3 model; rebuilds the env-style observation from live DB slots."""

    def __init__(self) -> None:
        self.model = None
        self._env_template = ParkingLotEnv()  # for zone/price arrays

    def load_model(self, path: Optional[str] = None) -> bool:
        path = path or str(config.MODEL_PATH)
        if not config.MODEL_PATH.exists() and not path.endswith(".zip"):
            return False
        try:
            # MaskablePPO first (trained with sb3-contrib), plain PPO fallback
            model = None
            try:
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(path)
            except Exception:
                from stable_baselines3 import PPO
                model = PPO.load(path)
            # refuse models trained on a different observation layout
            if model.observation_space.shape[0] != \
                    self._env_template.observation_space.shape[0]:
                print(f"[recommender] model obs dim mismatch — ignoring {path}")
                return False
            self.model = model
            return True
        except Exception:
            self.model = None
            return False

    @property
    def has_model(self) -> bool:
        return self.model is not None

    # -------------------------------------------------------------- heuristic
    def heuristic_pick(self, slots: list[dict], vehicle_type: str,
                       stay_minutes: float) -> Optional[dict]:
        allowed_zones = set(config.VEHICLE_ALLOWED_ZONES[vehicle_type])
        preferred_zone = {"ev": "E", "disabled": "D", "bike": "M"}.get(vehicle_type)
        free = [s for s in slots if s["status"] == "free" and s["zone"] in allowed_zones]
        if not free:
            return None

        def score(s: dict) -> tuple:
            # 1. preferred exact-match zone, 2. cheapest (keeps premium free), 3. lower index
            exact = 0 if (preferred_zone and s["zone"] == preferred_zone) else 1
            type_ok = 0 if config.ZONE_TYPE[s["zone"]] == (
                "ev" if vehicle_type == "ev" else
                "disabled" if vehicle_type == "disabled" else
                "bike" if vehicle_type == "bike" else "standard") else 1
            return (type_ok, exact, s["price_per_hour"], s["id"])

        return sorted(free, key=score)[0]

    # -------------------------------------------------------------- RL pick
    def rl_pick(self, slots: list[dict], vehicle_type: str, stay_minutes: float,
                sim_hour: float) -> tuple[Optional[dict], str]:
        """Returns (slot, mode) where mode is 'rl' or 'heuristic'."""
        if self.model is None:
            return self.heuristic_pick(slots, vehicle_type, stay_minutes), "heuristic"

        env_t = self._env_template
        n = env_t.n_slots
        status = np.full(n, 2, dtype=np.int64)  # default occupied (padding)
        # Mirror the real free-ratio of each zone onto the template slots:
        # the agent was trained on full-lot observations, so we reconstruct
        # an equivalent one from whatever slice of slots the DB exposes.
        for zone in np.unique(env_t.zone_of):
            idxs = np.where(env_t.zone_of == zone)[0]
            pool = [s for s in slots if s["zone"] == zone]
            frac_free = (len([s for s in pool if s["status"] == "free"]) / len(pool)) \
                if pool else 0.0
            n_free = int(round(frac_free * len(idxs)))
            status[idxs[:n_free]] = 0

        # build obs identical to env (status + vehicle + time + pressure + mask)
        obs = np.zeros(env_t.observation_space.shape, dtype=np.float32)
        obs[: n * 3] = np.eye(3, dtype=np.float32)[status].reshape(-1)
        off = n * 3
        if vehicle_type in VEHICLE_IDX:
            obs[off + VEHICLE_IDX[vehicle_type]] = 1.0
        off += len(VEHICLE_IDX)
        tb = int(np.clip(
            (sim_hour - config.SIM["start_hour"]) /
            max(config.SIM["close_hour"] - config.SIM["start_hour"], 1), 0, 0.999)
            * config.RL["obs_time_buckets"])
        obs[off + tb] = 1.0
        off += config.RL["obs_time_buckets"]
        free_ratio = float(np.mean(status == 0))
        obs[off] = 1.0 - free_ratio
        obs[off + 1] = 1.0 - free_ratio
        off += 2
        # allowed-slot mask for this vehicle (same layout as env observations)
        allowed = np.isin(env_t.zone_of, config.VEHICLE_ALLOWED_ZONES.get(vehicle_type, []))
        legal = (allowed & (status == 0)).astype(np.int8)
        obs[off: off + n] = legal.astype(np.float32)

        # mask inference to legal actions exactly as during training
        action_mask = np.append(legal, 1)  # reject is always allowed
        action, _ = self.model.predict(obs, deterministic=True,
                                       action_masks=action_mask)
        action = int(action)
        if action >= n:  # agent chose to reject
            return None, "rl"
        chosen_zone = env_t.zone_of[action]
        pool = [s for s in slots
                if s["zone"] == chosen_zone and s["status"] == "free"]
        if not pool:
            return self.heuristic_pick(slots, vehicle_type, stay_minutes), "heuristic"
        return pool[0], "rl"


# module-level singleton
recommender = SlotRecommender()
recommender.load_model()
