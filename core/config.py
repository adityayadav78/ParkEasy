"""Central configuration: lot layout, zones, pricing, RL & simulation settings."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "parkeasy.db"
MODEL_PATH = MODELS_DIR / "ppo_parking.zip"

for _d in (DATA_DIR, MODELS_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- lot layout
# Zones and per-zone slot counts. Slot types:
#   standard  -> regular cars
#   premium   -> near entrance, higher price
#   ev        -> EV charging, EVs only
#   disabled  -> accessible, disabled-permit only
#   bike      -> two-wheelers only
LOT_ID = "MAIN"
ZONES: dict[str, dict] = {
    "A": {"type": "standard", "slots": 24, "price_per_hour": 20.0},
    "B": {"type": "standard", "slots": 24, "price_per_hour": 20.0},
    "P": {"type": "premium",  "slots": 10, "price_per_hour": 40.0},
    "E": {"type": "ev",       "slots": 8,  "price_per_hour": 35.0},
    "D": {"type": "disabled", "slots": 4,  "price_per_hour": 15.0},
    "M": {"type": "bike",     "slots": 20, "price_per_hour": 8.0},
}

# Vehicle classes allowed per slot type
VEHICLE_ALLOWED_ZONES: dict[str, list[str]] = {
    "car":     ["A", "B", "P"],
    "ev":      ["A", "B", "P", "E"],   # EVs prefer E but may fall back
    "disabled": ["A", "B", "P", "D"],
    "bike":    ["M"],
}
ZONE_TYPE = {z: spec["type"] for z, spec in ZONES.items()}

# Reservation price multipliers by hour of day (demand curve)
HOURLY_MULTIPLIER: dict[int, float] = {
    0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.7,
    6: 0.9, 7: 1.1, 8: 1.5, 9: 1.8, 10: 1.8, 11: 1.7,
    12: 1.5, 13: 1.5, 14: 1.4, 15: 1.4, 16: 1.6, 17: 1.8,
    18: 1.9, 19: 1.6, 20: 1.3, 21: 1.0, 22: 0.8, 23: 0.6,
}

# ---------------------------------------------------------------- simulation
SIM = {
    "seconds_per_tick": 300,        # 5 simulated minutes per tick
    "start_hour": 6,                # lot opens 06:00
    "close_hour": 23,               # last entries until 23:00
    # arrivals per 5-min tick at given hours; interpolated linearly.
    # Peak ~3.0/tick = 36 arrivals/h × ~1.8 h mean stay ≈ 65-70 busy slots
    # against 80 car-class slots -> real capacity pressure at rush hours.
    "arrival_curve": {6: 0.30, 8: 3.00, 9: 2.40, 12: 1.60,
                      14: 2.00, 17: 2.80, 19: 2.00, 21: 0.80, 23: 0.0},
    "stay_minutes_mu": 110.0,       # lognormal mu
    "stay_minutes_sigma": 0.55,     # lognormal sigma
    "vehicle_mix": {                # probability of each class
        "car": 0.62, "ev": 0.08, "disabled": 0.05, "bike": 0.25,
    },
    "seed": 42,
}

# ---------------------------------------------------------------- RL / gym
RL = {
    "total_timesteps": 300_000,
    "n_envs": 4,
    "learning_rate": 3e-4,
    "gamma": 0.98,
    "obs_time_buckets": 8,          # time-of-day discretization
    "reward": {
        "assign": 1.0,              # base reward for successful assignment
        "reject_pressure": 0.0,     # scaled by remaining demand
        "reject_no_capacity": -1.0, # forced rejection (nothing available)
        "wrong_zone": -0.5,         # e.g. EV in standard slot (no charger)
        "premium_waste": -0.3,      # premium slot given to low-value stay
        "efficiency_bonus": 0.3,    # reward tight packing of standard slots
    },
}

# ---------------------------------------------------------------- misc
API_HOST = "0.0.0.0"
API_PORT = 8000
CURRENCY = "₹"
