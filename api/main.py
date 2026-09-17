"""ParkEasy FastAPI backend."""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import config
from core.database import (Booking, OccupancySnapshot, SessionLocal, Slot, User,
                           get_db, init_db)
from core.recommender import recommender
from core.simulator import simulator

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ---------------------------------------------------------------- schemas
class RecommendIn(BaseModel):
    vehicle_type: str = Field("car", pattern="^(car|ev|disabled|bike)$")
    stay_minutes: int = Field(60, ge=10, le=720)


class CheckinIn(BaseModel):
    user_id: int
    vehicle_type: str = Field("car", pattern="^(car|ev|disabled|bike)$")
    stay_minutes: int = Field(60, ge=10, le=720)


class CheckoutIn(BaseModel):
    booking_id: int


class UserIn(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    vehicle_plate: str
    vehicle_type: str = Field("car", pattern="^(car|ev|disabled|bike)$")


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    simulator.sim_hour = float(config.SIM["start_hour"])
    yield
    simulator.stop()


app = FastAPI(title="ParkEasy API", version="1.0.0",
              description="RL-optimized parking management",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """Serve the standalone HTML dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/health")
def health():
    return {"ok": True, "rl_model_loaded": recommender.has_model,
            "sim_hour": round(simulator.sim_hour, 2)}


# ---------------------------------------------------------------- slots
def _slot_dict(s: Slot) -> dict:
    return {"id": s.id, "code": s.code, "zone": s.zone, "slot_type": s.slot_type,
            "price_per_hour": s.price_per_hour, "status": s.status,
            "ev_charger": s.ev_charger}


@app.get("/api/v1/slots")
def list_slots(db=Depends(get_db)):
    slots = db.query(Slot).order_by(Slot.id).all()
    occ = sum(1 for s in slots if s.status == "occupied")
    return {"slots": [_slot_dict(s) for s in slots],
            "summary": {"total": len(slots), "occupied": occ,
                        "free": len(slots) - occ,
                        "occupancy_pct": round(100 * occ / max(len(slots), 1), 1)}}


@app.get("/api/v1/slots/{slot_id}")
def get_slot(slot_id: int, db=Depends(get_db)):
    slot = db.get(Slot, slot_id)
    if slot is None:
        raise HTTPException(404, "slot not found")
    return _slot_dict(slot)


# ---------------------------------------------------------------- recommend / book
@app.post("/api/v1/recommend")
def recommend(body: RecommendIn):
    db = SessionLocal()
    try:
        slots = [{"id": s.id, "code": s.code, "zone": s.zone,
                  "slot_type": s.slot_type, "price_per_hour": s.price_per_hour,
                  "status": s.status} for s in db.query(Slot).all()]
        slot, mode = recommender.rl_pick(
            slots, body.vehicle_type, body.stay_minutes, simulator.sim_hour)
        if slot is None:
            raise HTTPException(409, "No suitable slot available — try later")
        hour = int(simulator.sim_hour) % 24
        mult = config.HOURLY_MULTIPLIER[hour]
        return {"slot": slot, "assigned_by": mode,
                "estimated_price": round(slot["price_per_hour"] * mult *
                                         (body.stay_minutes / 60.0), 2),
                "sim_hour": round(simulator.sim_hour, 2)}
    finally:
        db.close()


@app.post("/api/v1/checkin")
def checkin(body: CheckinIn):
    db = SessionLocal()
    try:
        slots = [{"id": s.id, "code": s.code, "zone": s.zone,
                  "slot_type": s.slot_type, "price_per_hour": s.price_per_hour,
                  "status": s.status} for s in db.query(Slot).all()]
        slot_dict, mode = recommender.rl_pick(
            slots, body.vehicle_type, body.stay_minutes, simulator.sim_hour)
        if slot_dict is None:
            raise HTTPException(409, "Lot full for this vehicle type")
        slot = db.get(Slot, slot_dict["id"])
        if slot is None or slot.status != "free":
            raise HTTPException(409, "Slot just got taken — retry")
        slot.status = "occupied"
        now = dt.datetime.now(dt.timezone.utc)
        hour = int(simulator.sim_hour) % 24
        cost = round(slot.price_per_hour * config.HOURLY_MULTIPLIER[hour] *
                     (body.stay_minutes / 60.0), 2)
        booking = Booking(
            user_id=body.user_id, slot_id=slot.id,
            vehicle_type=body.vehicle_type, start_time=now, status="active",
            cost=cost, assigned_by=mode,
            meta={"planned_end_sim_hour":
                  simulator.sim_hour + body.stay_minutes / 60.0,
                  "stay_minutes": body.stay_minutes})
        db.add(booking)
        db.commit()
        return {"booking_id": booking.id, "slot": _slot_dict(slot),
                "assigned_by": mode, "cost": cost,
                "message": f"Assigned {slot.code} — {config.CURRENCY}{cost:.0f} est."}
    finally:
        db.close()


@app.post("/api/v1/checkout")
def checkout(body: CheckoutIn):
    db = SessionLocal()
    try:
        booking = db.get(Booking, body.booking_id)
        if booking is None or booking.status != "active":
            raise HTTPException(404, "active booking not found")
        booking.status = "completed"
        booking.end_time = dt.datetime.now(dt.timezone.utc)
        slot = db.get(Slot, booking.slot_id)
        if slot:
            slot.status = "free"
        db.commit()
        return {"booking_id": booking.id, "final_cost": booking.cost,
                "duration_min": booking.meta.get("stay_minutes"),
                "message": "Checked out — slot released"}
    finally:
        db.close()


# ---------------------------------------------------------------- analytics
@app.get("/api/v1/occupancy/history")
def occupancy_history(limit: int = 200, db=Depends(get_db)):
    rows = db.query(OccupancySnapshot).order_by(
        OccupancySnapshot.id.desc()).limit(limit).all()
    rows = list(reversed(rows))
    return {"history": [
        {"ts": r.ts.isoformat(), "sim_hour": r.sim_hour, "occupied": r.occupied,
         "reserved": r.reserved, "total": r.total_slots, "revenue": r.revenue,
         "rejections": r.rejections} for r in rows]}


def _count(db, model, *filters) -> int:
    q = db.query(model.id)
    for f in filters:
        q = q.filter(f)
    return q.count()


@app.get("/api/v1/stats")
def stats(db=Depends(get_db)):
    total = _count(db, Slot)
    occupied = _count(db, Slot, Slot.status == "occupied")
    revenue = sum(
        (b.cost or 0.0) for b in db.query(Booking).filter(
            Booking.status.in_(["active", "completed"])).all())
    active = _count(db, Booking, Booking.status == "active")
    completed = _count(db, Booking, Booking.status == "completed")
    by_zone = {}
    for zone, in db.query(Slot.zone).distinct():
        zs = db.query(Slot).filter(Slot.zone == zone).all()
        occ = sum(1 for s in zs if s.status == "occupied")
        by_zone[zone] = {"total": len(zs), "occupied": occ,
                         "occupancy_pct": round(100 * occ / max(len(zs), 1), 1)}
    return {"total_slots": total, "occupied": occupied,
            "occupancy_pct": round(100 * occupied / max(total, 1), 1),
            "revenue": round(revenue, 2), "active_bookings": active,
            "completed_bookings": completed,
            "sim_rejections": simulator.total_rejections,
            "sim_hour": round(simulator.sim_hour, 2),
            "by_zone": by_zone}


# ---------------------------------------------------------------- users
@app.post("/api/v1/users")
def create_user(body: UserIn, db=Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(409, "email already registered")
    user = User(**body.model_dump())
    db.add(user)
    db.commit()
    return {"id": user.id, "name": user.name, "email": user.email}


@app.get("/api/v1/users")
def list_users(db=Depends(get_db)):
    users = db.query(User).order_by(User.id).all()
    return {"users": [
        {"id": u.id, "name": u.name, "email": u.email,
         "vehicle_plate": u.vehicle_plate, "vehicle_type": u.vehicle_type}
        for u in users]}


# ---------------------------------------------------------------- bookings
@app.get("/api/v1/bookings/active")
def active_bookings(db=Depends(get_db)):
    rows = db.query(Booking).filter(Booking.status == "active").order_by(
        Booking.id.desc()).limit(100).all()
    out = []
    for b in rows:
        slot = db.get(Slot, b.slot_id)
        user = db.get(User, b.user_id)
        out.append({
            "booking_id": b.id, "slot": slot.code if slot else "?",
            "user": user.name if user else "?", "plate": user.vehicle_plate if user else "?",
            "vehicle_type": b.vehicle_type, "assigned_by": b.assigned_by,
            "cost": b.cost, "planned_end_sim_hour": round(
                (b.meta or {}).get("planned_end_sim_hour", 0), 2)})
    return {"bookings": out}


# ---------------------------------------------------------------- admin
@app.post("/api/v1/admin/train")
def admin_train(steps: int = 60_000):
    """Retrain the RL agent synchronously (small budgets only)."""
    from rl.train import train
    try:
        train(total_timesteps=max(10_000, min(steps, 200_000)))
        recommender.load_model()
        return {"ok": True, "message": f"Model retrained and reloaded ({recommender.has_model=})"}
    except Exception as exc:
        raise HTTPException(500, f"training failed: {exc}")


@app.post("/api/v1/admin/reset")
def admin_reset(db=Depends(get_db)):
    """Release all slots and end all active bookings."""
    db.query(Booking).filter(Booking.status == "active").update(
        {"status": "completed", "end_time": dt.datetime.now(dt.timezone.utc)})
    db.query(Slot).update({"status": "free"})
    db.commit()
    return {"ok": True, "message": "Lot reset"}


# ---------------------------------------------------------------- simulation control
@app.post("/api/v1/sim/tick")
def sim_tick(n: int = 1):
    out = [simulator.tick() for _ in range(max(1, min(n, 20)))]
    return {"ticks": out, "sim_hour": round(simulator.sim_hour, 2)}


@app.post("/api/v1/sim/start")
def sim_start(interval: float = 2.0):
    simulator.start(interval=interval)
    return {"running": simulator.running, "interval": interval}


@app.post("/api/v1/sim/stop")
def sim_stop():
    simulator.stop()
    return {"running": simulator.running}


@app.get("/api/v1/sim/state")
def sim_state():
    return {"running": simulator.running, "ticks": simulator.ticks,
            "sim_hour": round(simulator.sim_hour, 2),
            "rejections": simulator.total_rejections,
            "revenue": round(simulator.session_revenue, 2)}
