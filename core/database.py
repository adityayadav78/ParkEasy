"""SQLAlchemy models + engine/session management (SQLite by default)."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String,
                        create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)

from core import config

engine = create_engine(
    f"sqlite:///{config.DB_PATH}",
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(160), unique=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    vehicle_plate: Mapped[str] = mapped_column(String(20))
    vehicle_type: Mapped[str] = mapped_column(String(12), default="car")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))

    bookings: Mapped[list["Booking"]] = relationship(back_populates="user")


class Slot(Base):
    __tablename__ = "slots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True)   # e.g. A-07
    zone: Mapped[str] = mapped_column(String(2))                 # A / B / P / E / D / M
    section: Mapped[str] = mapped_column(String(12), default="car")  # car/bike/ev/disabled
    slot_type: Mapped[str] = mapped_column(String(12))           # standard/premium/ev/disabled/bike
    price_per_hour: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(12), default="free")  # free/reserved/occupied/maintenance
    ev_charger: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))

    bookings: Mapped[list["Booking"]] = relationship(back_populates="slot")


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    slot_id: Mapped[int] = mapped_column(ForeignKey("slots.id"))
    vehicle_type: Mapped[str] = mapped_column(String(12))
    start_time: Mapped[dt.datetime] = mapped_column(DateTime)
    end_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(12), default="active")  # active/completed/cancelled
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    assigned_by: Mapped[str] = mapped_column(String(12), default="heuristic")  # rl/heuristic
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    user: Mapped["User"] = relationship(back_populates="bookings")
    slot: Mapped["Slot"] = relationship(back_populates="bookings")


class OccupancySnapshot(Base):
    """Periodic snapshot of lot state, used for history charts."""
    __tablename__ = "occupancy_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    sim_hour: Mapped[float] = mapped_column(Float)
    total_slots: Mapped[int] = mapped_column(Integer)
    occupied: Mapped[int] = mapped_column(Integer)
    reserved: Mapped[int] = mapped_column(Integer)
    revenue: Mapped[float] = mapped_column(Float)
    rejections: Mapped[int] = mapped_column(Integer)


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency: yield a session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
