"""Gymnasium environment for dynamic parking-slot allocation.

Episode  = one simulated operating day (arrivals stop at closing hour).
Step     = one vehicle arrival; the agent chooses a slot to assign (or reject).
Observation = per-slot status + arriving vehicle class + time bucket + pressure.
Action   = Discrete(n_slots + 1): index -> assign that slot, n_slots -> reject.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core import config

STATUS_IDX = {"free": 0, "reserved": 1, "occupied": 2}
VEHICLE_IDX = {"car": 0, "ev": 1, "disabled": 2, "bike": 3}


def _interp_curve(curve: dict[int, float], hour: float) -> float:
    """Linearly interpolate an hourly curve at fractional hour."""
    xs = sorted(curve)
    if hour <= xs[0]:
        return curve[xs[0]]
    if hour >= xs[-1]:
        return curve[xs[-1]]
    for x0, x1 in zip(xs, xs[1:]):
        if x0 <= hour <= x1:
            f = (hour - x0) / (x1 - x0) if x1 > x0 else 0.0
            return curve[x0] + f * (curve[x1] - curve[x0])
    return 0.0


class ParkingLotEnv(gym.Env):
    """Dynamic slot-allocation environment.

    The agent sees each arriving vehicle and must decide which slot to give it
    (or reject it). Constrained slots (EV / disabled) are scarce: handing a
    charger slot to a plain car can force a later rejection of a true EV — the
    agent must learn this long-horizon trade-off.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_days: int = 1,
        seed: Optional[int] = None,
        slots: Optional[list[dict]] = None,
        reward_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.reward_cfg = {**config.RL["reward"], **(reward_cfg or {})}

        # ---- slot table -------------------------------------------------
        if slots is None:
            slots = []
            for zone, spec in config.ZONES.items():
                for i in range(1, spec["slots"] + 1):
                    slots.append({
                        "code": f"{zone}-{i:02d}",
                        "zone": zone,
                        "slot_type": spec["type"],
                        "price_per_hour": spec["price_per_hour"],
                    })
        self.slots: list[dict] = slots
        self.n_slots = len(self.slots)
        self.zone_of = np.array([s["zone"] for s in self.slots])
        self.price_of = np.array([s["price_per_hour"] for s in self.slots], dtype=np.float32)

        # ---- spaces ------------------------------------------------------
        n_status, n_veh = len(STATUS_IDX), len(VEHICLE_IDX)
        n_time = config.RL["obs_time_buckets"]
        # status one-hots + vehicle + time + pressure(2) + allowed-slot mask
        obs_dim = self.n_slots * n_status + n_veh + n_time + 2 + self.n_slots
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_slots + 1)  # last action = reject

        self.n_time_buckets = n_time
        self.day_minutes = (config.SIM["close_hour"] - config.SIM["start_hour"]) * 60
        self.n_days = n_days

        # ---- episode state ------------------------------------------------
        self.reset()

    # ------------------------------------------------------------------ utils
    def _allowed_mask(self, vehicle: str) -> np.ndarray:
        zones = config.VEHICLE_ALLOWED_ZONES[vehicle]
        return np.isin(self.zone_of, zones)

    def _time_bucket(self) -> int:
        frac = (self.sim_minutes - config.SIM["start_hour"] * 60) / max(self.day_minutes, 1)
        return int(np.clip(frac, 0, 0.999) * self.n_time_buckets)

    def _pressure(self, vehicle: str) -> tuple[float, float]:
        free = self.status == 0
        allowed = free & self._allowed_mask(vehicle)
        return (
            1.0 - float(free.mean()),
            1.0 - float(allowed.mean()),
        )

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        # one-hot status per slot
        obs[: self.n_slots * 3] = np.eye(3, dtype=np.float32)[self.status].reshape(-1)
        off = self.n_slots * 3
        obs[off + VEHICLE_IDX[self.cur_vehicle]] = 1.0
        off += len(VEHICLE_IDX)
        obs[off + self._time_bucket()] = 1.0
        off += self.n_time_buckets
        press, press_allowed = self._pressure(self.cur_vehicle)
        obs[off], obs[off + 1] = press, press_allowed
        off += 2
        # allowed-slot mask: which actions are legal right now (big learning aid)
        obs[off: off + self.n_slots] = self._valid_actions()[: self.n_slots]
        return obs

    def _next_arrival(self) -> float:
        """Simulated minutes until the next arrival from current sim time."""
        hour = config.SIM["start_hour"] * 60 + self.sim_minutes
        hour /= 60.0
        lam = _interp_curve(config.SIM["arrival_curve"], hour)  # arrivals per 5-min tick
        if lam <= 0:
            return float("inf")
        gap_ticks = self.rng.exponential(1.0 / lam)
        return max(gap_ticks, 0.2) * config.SIM["seconds_per_tick"] / 60.0

    def _sample_vehicle(self) -> tuple[str, float]:
        types = list(config.SIM["vehicle_mix"])
        p = np.array([config.SIM["vehicle_mix"][t] for t in types])
        veh = self.rng.choice(types, p=p / p.sum())
        # lognormal over log-minutes: median stay ~ SIM["stay_minutes_mu"]
        mu = float(np.log(config.SIM["stay_minutes_mu"]))
        stay = float(np.clip(self.rng.lognormal(mu, config.SIM["stay_minutes_sigma"]),
                             15, 480))
        return str(veh), stay

    def _hour(self) -> float:
        return (config.SIM["start_hour"] * 60 + self.sim_minutes) / 60.0

    def _slot_price(self, slot_idx: int, stay_minutes: float) -> float:
        hour = int(np.clip(self._hour(), 0, 23))
        mult = config.HOURLY_MULTIPLIER[hour]
        return float(self.price_of[slot_idx] * mult * (stay_minutes / 60.0))

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.status = np.zeros(self.n_slots, dtype=np.int64)   # all free
        self.dep_times = np.full(self.n_slots, np.inf)          # departure sim-minutes
        self.sim_minutes = 0.0
        self.day_idx = 0
        self.stats = {"arrivals": 0, "accepted": 0, "rejected": 0,
                      "invalid": 0, "revenue": 0.0, "wrong_zone": 0}
        self._schedule_next()
        self._advance_departures()
        self.cur_vehicle, self.cur_stay = self._sample_vehicle()
        return self._build_obs(), self._info()

    def _schedule_next(self) -> None:
        self.next_gap = self._next_arrival()

    def _advance_departures(self) -> None:
        due = (self.status == 2) & (self.dep_times <= self.sim_minutes + 1e-9)
        self.status[due] = 0
        self.dep_times[due] = np.inf

    def _info(self) -> dict:
        return {
            "stats": dict(self.stats),
            "sim_minutes": self.sim_minutes,
            "vehicle": self.cur_vehicle,
            "stay_minutes": self.cur_stay,
            "valid_actions": self._valid_actions(),
        }

    def _valid_actions(self) -> np.ndarray:
        mask = (self.status == 0) & self._allowed_mask(self.cur_vehicle)
        actions = np.zeros(self.n_slots + 1, dtype=bool)
        actions[: self.n_slots] = mask
        actions[-1] = True  # reject is always legal
        return actions

    def action_masks(self) -> np.ndarray:
        """MaskablePPO hook: 1 = legal action this step."""
        return self._valid_actions().astype(np.int8)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"bad action {action}"
        veh, stay = self.cur_vehicle, self.cur_stay
        r = self.reward_cfg
        reward = 0.0
        valid = self._valid_actions()

        if action == self.n_slots:  # ---- voluntary rejection
            self.stats["rejected"] += 1
            any_capacity = bool(valid[:-1].any())
            pressure, _ = self._pressure(veh)
            if not any_capacity:
                reward = r["reject_no_capacity"]      # forced: lot is full
            elif pressure > 0.75:
                reward = r["reject_pressure"]         # nearly full: borderline
            else:
                reward = -2.0                          # wasting open capacity
        elif not valid[action]:      # ---- illegal assignment attempt
            self.stats["invalid"] += 1
            self.stats["rejected"] += 1
            reward = -1.5
        else:                        # ---- legal assignment
            self.status[action] = 2
            self.dep_times[action] = self.sim_minutes + stay
            price = self._slot_price(action, stay)
            self.stats["accepted"] += 1
            self.stats["revenue"] += price
            reward = r["assign"] + price / 50.0

            zone = self.zone_of[action]
            ztype = config.ZONE_TYPE[zone]
            # wrong-zone penalties (capacity misuse)
            if veh == "ev" and ztype != "ev":
                reward += r["wrong_zone"]
                self.stats["wrong_zone"] += 1
            elif veh == "disabled" and ztype != "disabled":
                reward += r["wrong_zone"] * 0.4
                self.stats["wrong_zone"] += 1
            elif veh != "ev" and ztype == "ev":
                reward += r["wrong_zone"]
                self.stats["wrong_zone"] += 1
            # premium slot wasted on a very short stay
            if ztype == "premium" and stay < 45:
                reward += r["premium_waste"]
            # efficiency bonus: tight packing inside standard zones
            if ztype == "standard":
                idxs = np.where(self.zone_of == zone)[0]
                lowest_free = idxs[self.status[idxs] == 0]
                if len(lowest_free) and action == lowest_free[0]:
                    reward += r["efficiency_bonus"] * 0.5

        self.stats["arrivals"] += 1

        # ---- advance time to next arrival --------------------------------
        self.sim_minutes += self.next_gap
        terminated = self.sim_minutes > self.day_minutes
        if not terminated:
            self._advance_departures()
            self._schedule_next()
            self.cur_vehicle, self.cur_stay = self._sample_vehicle()
        truncated = False
        return self._build_obs(), float(reward), terminated, truncated, self._info()

    def render(self) -> str:
        rows = []
        for zone in config.ZONES:
            idxs = np.where(self.zone_of == zone)[0]
            cells = "".join(
                {0: "[ ]", 1: "[R]", 2: "[X]"}[int(self.status[i])] for i in idxs)
            rows.append(f"{zone} ({config.ZONE_TYPE[zone]:8s}): {cells}")
        hour = self._hour()
        return (f"--- Day {self.day_idx + 1} {int(hour):02d}:{int(hour % 1 * 60):02d} "
                f"occ={int((self.status == 2).sum())}/{self.n_slots} ---\n" + "\n".join(rows))


# Register so `gym.make("ParkingLot-v0")` works
gym.register(id="ParkingLot-v0", entry_point=ParkingLotEnv, max_episode_steps=500)
