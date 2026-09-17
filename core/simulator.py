"""Digital-twin simulator: continuously generates arrivals/departures on live DB slots."""
from __future__ import annotations

import datetime as dt
import math
import threading
from typing import Optional

import numpy as np

from core import config
from core.database import Booking, OccupancySnapshot, SessionLocal, Slot, User
from core.recommender import recommender

_lock = threading.Lock()


class LotSimulator:
    """Drives the live database forward in simulated time."""

    def __init__(self) -> None:
        self.sim_hour: float = float(config.SIM["start_hour"])
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.ticks = 0
        self.total_rejections = 0
        self.session_revenue = 0.0
        self.auto_users: list[dict] = []

    # ------------------------------------------------------------- helpers
    def _ensure_users(self, db) -> None:
        if self.auto_users:
            return
        users = db.query(User).all()
        if users:
            self.auto_users = [{"id": u.id, "plate": u.vehicle_plate,
                                "type": u.vehicle_type} for u in users]

    def _random_vehicle(self) -> dict:
        import random
        mix = config.SIM["vehicle_mix"]
        veh = random.choices(list(mix), weights=list(mix.values()))[0]
        stay = float(np.clip(random.lognormvariate(
            math.log(config.SIM["stay_minutes_mu"]), config.SIM["stay_minutes_sigma"]),
            15, 480))
        return {"vehicle_type": veh, "stay_minutes": stay}

    def _price_for(self, slot: Slot, stay_minutes: float) -> float:
        hour = int(self.sim_hour) % 24
        mult = config.HOURLY_MULTIPLIER[hour]
        return round(slot.price_per_hour * mult * (stay_minutes / 60.0), 2)

    def _departures(self, db) -> int:
        now = dt.datetime.now(dt.timezone.utc)
        active = db.query(Booking).filter(Booking.status == "active").all()
        ended = 0
        for b in active:
            end = (b.meta or {}).get("planned_end_sim_hour")
            if end is not None and self.sim_hour >= end:
                b.status = "completed"
                b.end_time = now
                slot = db.get(Slot, b.slot_id)
                if slot:
                    slot.status = "free"
                ended += 1
        return ended

    # ------------------------------------------------------------- one tick
    def tick(self, n_arrivals: Optional[int] = None) -> dict:
        """Advance one simulation tick: departures, arrivals, snapshot."""
        with _lock:
            db = SessionLocal()
            try:
                stats = {"departed": self._departures(db), "arrived": 0,
                         "rejected": 0}

                lam = self._arrival_rate()
                expected = lam if n_arrivals is None else float(n_arrivals)
                import random
                n = int(random.random() < expected % 1) + int(expected) \
                    if n_arrivals is None else n_arrivals
                for _ in range(max(n, 0)):
                    veh = self._random_vehicle()
                    self._ensure_users(db)
                    user = random.choice(self.auto_users) if self.auto_users else None
                    slots = [{"id": s.id, "code": s.code, "zone": s.zone,
                              "slot_type": s.slot_type,
                              "price_per_hour": s.price_per_hour,
                              "status": s.status} for s in db.query(Slot).all()]
                    slot_dict, mode = recommender.rl_pick(
                        slots, veh["vehicle_type"], veh["stay_minutes"], self.sim_hour)
                    if slot_dict is None:
                        stats["rejected"] += 1
                        self.total_rejections += 1
                        continue
                    slot = db.get(Slot, slot_dict["id"])
                    if slot is None or slot.status != "free":
                        stats["rejected"] += 1
                        continue
                    slot.status = "occupied"
                    now = dt.datetime.now(dt.timezone.utc)
                    cost = self._price_for(slot, veh["stay_minutes"])
                    planned_end = self.sim_hour + veh["stay_minutes"] / 60.0
                    booking = Booking(
                        user_id=user["id"] if user else 1,
                        slot_id=slot.id,
                        vehicle_type=veh["vehicle_type"],
                        start_time=now,
                        status="active",
                        cost=cost,
                        assigned_by=mode,
                        meta={"planned_end_sim_hour": planned_end,
                              "stay_minutes": veh["stay_minutes"]},
                    )
                    db.add(booking)
                    self.session_revenue += cost
                    stats["arrived"] += 1

                self._snapshot(db)
                db.commit()
                # advance sim clock by one tick (5 min) — wrap after closing
                self.sim_hour += config.SIM["seconds_per_tick"] / 3600.0
                if self.sim_hour >= config.SIM["close_hour"] + 1:
                    self.sim_hour = float(config.SIM["start_hour"])
                self.ticks += 1
                return stats
            finally:
                db.close()

    def _arrival_rate(self) -> float:
        curve = config.SIM["arrival_curve"]
        xs = sorted(curve)
        h = self.sim_hour
        if h <= xs[0]:
            return curve[xs[0]]
        if h >= xs[-1]:
            return curve[xs[-1]]
        for x0, x1 in zip(xs, xs[1:]):
            if x0 <= h <= x1:
                f = (h - x0) / (x1 - x0) if x1 > x0 else 0
                return curve[x0] + f * (curve[x1] - curve[x0])
        return 0.0

    def _snapshot(self, db) -> None:
        slots = db.query(Slot).all()
        occupied = sum(1 for s in slots if s.status == "occupied")
        reserved = sum(1 for s in slots if s.status == "reserved")
        db.add(OccupancySnapshot(
            sim_hour=self.sim_hour,
            total_slots=len(slots),
            occupied=occupied,
            reserved=reserved,
            revenue=round(self.session_revenue, 2),
            rejections=self.total_rejections,
        ))
        # keep table small
        if db.query(OccupancySnapshot).count() > 2000:
            oldest = db.query(OccupancySnapshot).order_by(
                OccupancySnapshot.id).first()
            if oldest:
                db.delete(oldest)

    # ------------------------------------------------------------- loop
    def start(self, interval: float = 2.0) -> None:
        if self.running:
            return
        self.running = True

        def loop() -> None:
            import time
            while self.running:
                try:
                    self.tick()
                except Exception as exc:  # keep the twin alive
                    print(f"[simulator] tick error: {exc}")
                time.sleep(interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False


simulator = LotSimulator()
