"""Create schema + seed users and slots."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config                                   # noqa: E402
from core.database import SessionLocal, init_db            # noqa: E402
from core.database import Slot, User                       # noqa: E402


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        if db.query(Slot).count() > 0:
            print("Database already seeded — nothing to do.")
            return

        for zone, spec in config.ZONES.items():
            for i in range(1, spec["slots"] + 1):
                db.add(Slot(
                    code=f"{zone}-{i:02d}", zone=zone,
                    slot_type=spec["type"],
                    price_per_hour=spec["price_per_hour"],
                    ev_charger=(spec["type"] == "ev")))

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
