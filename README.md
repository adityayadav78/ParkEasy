# ParkEasy 🚗🅿️

**An intelligent, RL-driven parking management system.** ParkEasy replaces static parking lots
with a dynamically optimized allocation engine: a Reinforcement Learning agent learns when to
*reserve* slots and when to keep them *open*, while a simulation models realistic arrival
traffic, stay durations, and occupancy patterns. A FastAPI backend exposes live slot state and
smart recommendations; a Streamlit dashboard visualizes the lot in real time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gymnasium](https://img.shields.io/badge/RL-Gymnasium%20%2B%20SB3%20(PPO)-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![DB](https://img.shields.io/badge/DB-SQLite-lightgrey)

---

## Features

- **RL Parking Gym** (`parking_gym`) — a Gymnasium environment simulating a multi-slot parking lot
  with Poisson arrivals, stay durations, hourly demand curves, and vehicle-type constraints
  (car / EV / disabled / bike).
- **PPO agent** (Stable-Baselines3) learns an **allocation policy**: accept → assign a slot,
  reject → protect capacity for expected higher-value arrivals.
- **Rejection-rate evaluation**: the trained agent is compared against greedy
  first-fit / nearest-first baselines.
- **FastAPI backend**: live slot map, check-in / check-out flow, occupancy time-series,
  recommendation endpoint, and an auto-refreshing simulation loop ("digital twin").
- **Streamlit dashboard**: 3-lane lot grid with color-coded slot cards, KPI tiles
  (occupancy %, revenue, rejections), occupancy trend chart, revenue chart, and an
  **Admin** page to retrain the agent and adjust the simulator.
- **SQLite persistence**: users, bookings, parking sessions, occupancy history —
  swappable to PostgreSQL via SQLAlchemy.

## Project structure

```
parkeasy/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt
├── README.md
├── api/
│   └── main.py                 # FastAPI app + standalone HTML dashboard at /
├── core/
│   ├── __init__.py
│   ├── config.py               # Lots, zones, pricing, simulation settings
│   ├── database.py             # SQLAlchemy engine/session + ORM models
│   ├── simulator.py            # Continuous arrival/departure simulation (digital twin)
│   └── recommender.py          # Slot recommendation logic (RL policy + heuristic fallback)
├── envs/
│   ├── __init__.py
│   └── parking_gym_env.py      # Gymnasium environment (MaskablePPO-compatible)
├── rl/
│   ├── __init__.py
│   ├── train.py                # MaskablePPO training + evaluation
│   └── baselines.py            # Greedy heuristics for comparison
├── static/                     # HTML/CSS/JS front-end (served by FastAPI)
├── data/                       # SQLite database lives here (gitignored)
├── models/                     # Trained SB3 models (gitignored)
└── scripts/
    ├── init_db.py              # Create schema + seed data
    └── train_agent.py          # CLI: train & evaluate the agent
```

## Quickstart

### 1. Install

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash)
# source .venv/bin/activate          # Linux/macOS
pip install -r requirements.txt
```

### 2. Create & seed the database

```bash
python scripts/init_db.py
```

### 3. Train the RL agent (or use the built-in heuristic fallback)

```bash
python scripts/train_agent.py --timesteps 100000
```

Trained model is saved to `models/ppo_parking.zip`. The API and dashboard fall back
to a nearest-fit heuristic automatically when no model is present.

### 4. Run the API (terminal 1)

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### 5. Run the dashboard (terminal 2)

```bash
streamlit run app.py
```

Open http://localhost:8501 — the dashboard talks to the API at `http://localhost:8000`.

## How the RL environment works

| Element | Value |
|---|---|
| Observation | Per-slot status (`free/reserved/occupied`) + vehicle class + time-of-day bucket + remaining demand pressure |
| Action | `0..N-1` assign slot `i`, `N` = reject request |
| Reward | Slot price per class − (reject when capacity later goes unused) − (assign premium slot to wrong class) − overstay penalty |
| Episode | One simulated operating day (arrival process stops at `T_close`) |

The agent learns *opportunistic capacity control*: early in the day, when demand pressure
is low, it reserves only cheap slots and keeps premium/EV slots open for higher-paying,
more constrained arrivals later.

### Measured results (300k timesteps, 5 evaluation days)

| Policy | Rejection rate | Daily revenue | Wrong-zone assignments |
|---|---|---|---|
| **RL (Maskable PPO)** | **0.0 %** | **₹25,416** | 10.0 |
| nearest_fit (greedy) | 0.18 % | ₹20,457 | 1.8 |
| first_fit (greedy) | 0.57 % | ₹19,587 | 43.8 |

The trained agent accepts **every** feasible arrival (zero rejections) while earning
**~24 % more revenue** than the best greedy baseline — it packs standard slots tightly
and steers EVs/disabled drivers to their dedicated zones instead of letting plain cars
scarf charger/accessible slots.

> Uses `sb3-contrib`'s **MaskablePPO** with invalid-action masking: illegal moves
> (occupied / wrong-class slots) are masked out at both training and inference time,
> which is what makes the policy actually learn slot choice instead of action legality.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/slots` | All slots + live status |
| GET | `/api/v1/slots/{id}` | Single slot |
| POST | `/api/v1/recommend` | Best slot for a vehicle + expected price |
| POST | `/api/v1/checkin` | Assign slot, create booking + session |
| POST | `/api/v1/checkout` | Close session, compute cost |
| GET | `/api/v1/occupancy/history` | Time-series of occupancy |
| GET | `/api/v1/stats` | KPIs (occupancy, revenue, rejections) |
| POST | `/api/v1/sim/tick` | Advance the digital-twin simulation |
| POST | `/api/v1/sim/start` \| `/api/v1/sim/stop` | Auto-run simulation loop |
| GET | `/api/v1/health` | Liveness |

## Configuration

Everything tunable lives in `core/config.py`: lot layout, zones (standard / premium /
EV / disabled / bike), hourly price multipliers, arrival-rate curves, and pricing.
