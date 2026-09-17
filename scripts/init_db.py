"""Create schema + seed users and slots (with section-aware migration)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import inspect, text                    # noqa: E402

from core import config                                  # noqa: E402
from core.database import SessionLocal, init_db          # noqa: E402
from core.database import Slot, User                     # noqa: E402


def _migrate_sections(db) -> None:
    """Add + backfill the `slots.section` column on pre-section databases."""
    inspector = inspect(db.bind)
    cols = {c["name"] for c in inspector.get_columns("slots")}
    if "section" not in cols:
        db.execute(text(
            "ALTER TABLE slots ADD COLUMN section VARCHAR(12) DEFAULT 'car'"))
        db.commit()
        print("Migrated: added slots.section column.")
    # zone->section mapping is authoritative; backfill unconditionally
    # (SQLite reports DEFAULT 'car' for pre-existing rows, so value-based
    # filters like IS NULL would never match)
    for zone, section in config.ZONE_SECTION.items():
        db.query(Slot).filter(Slot.zone == zone).update(
            {"section": section})
    db.commit()


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        # schema-level migration MUST run before any ORM query on Slot
        _migrate_sections(db)
        if db.query(Slot).count() == 0:
            for zone, spec in config.ZONES.items():
                for i in range(1, spec["slots"] + 1):
                    db.add(Slot(
                        code=f"{zone}-{i:02d}", zone=zone,
                        section=config.ZONE_SECTION[zone],
                        slot_type=spec["type"],
                        price_per_hour=spec["price_per_hour"],
                        ev_charger=(spec["type"] == "ev")))

        if db.query(User).count() == 0:
            users = [
                ("Aarav Sharma", "aarav@example.com", "9876500011", "KA01AB1234", "car"),
                ("Diya Patel", "diya@example.com", "9876500022", "KA05MP8890", "ev"),
                ("Rohan Verma", "rohan@example.com", "9876500033", "KA03CD5678", "car"),
                ("Meera Iyer", "meera@example.com", "9876500044", "KA01XY4321", "disabled"),
                ("Kabir Singh", "kabir@example.com", "9876500055", "KA09KL7788", "bike"),
            ]
            for name, email, phone, plate, vtype in users:
                db.add(User(name=name, email=email, phone=phone,
                            vehicle_plate=plate, vehicle_type=vtype))

        db.commit()
        n_slots = db.query(Slot).count()
        n_users = db.query(User).count()
        print(f"Seeded {n_slots} slots and {n_users} users into {config.DB_PATH}")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
