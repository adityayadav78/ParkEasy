"""ParkEasy — RL-optimized parking management dashboard."""
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://localhost:8000"
REFRESH_KEY = "auto_refresh"

st.set_page_config(page_title="ParkEasy", page_icon="🅿️", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------- api helpers
def api_get(path: str, default=None):
    try:
        r = requests.get(f"{API}{path}", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_post(path: str, payload: dict | None = None):
    try:
        r = requests.post(f"{API}{path}", json=payload or {}, timeout=5)
        if r.status_code >= 400:
            try:
                return {"_error": r.json().get("detail", r.text)}
            except Exception:
                return {"_error": r.text}
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------- styling
STATUS_COLORS = {"free": "#22c55e", "occupied": "#ef4444",
                 "reserved": "#f59e0b", "maintenance": "#94a3b8"}
ZONE_ICONS = {"standard": "🚗", "premium": "⭐", "ev": "⚡",
              "disabled": "♿", "bike": "🛵"}
SECTION_META = {
    "car":      {"name": "Car Section",                "icon": "🚗"},
    "bike":     {"name": "Bike / Two-Wheeler Section", "icon": "🛵"},
    "ev":       {"name": "EV Charging Section",        "icon": "⚡"},
    "disabled": {"name": "Accessible Section",         "icon": "♿"},
}

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px; padding: 16px; border: 1px solid #475569;
    }
    .slotcard {
        border-radius: 8px; padding: 6px 4px; text-align: center;
        font-weight: 700; font-size: 0.78rem; color: #0f172a;
        border: 2px solid rgba(0,0,0,0.25);
    }
    .lotmap { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
        gap: 6px; }
    .lotlane { margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🅿️ ParkEasy")
    st.caption("RL-optimized parking management")

    health = api_get("/api/v1/health")
    if health:
        st.success(f"API online · RL model: {'✅' if health.get('rl_model_loaded') else 'heuristic'}"
                   f" · sim {health.get('sim_hour', 0):04.1f}h")
    else:
        st.error("API offline — start it with:\n`uvicorn api.main:app --port 8000`")

    st.divider()
    page = st.radio("Navigation", ["Live Lot", "Analytics", "Book Parking", "Admin"],
                    horizontal=False)

    st.divider()
    sim_state = api_get("/api/v1/sim/state", {})
    auto = st.toggle("Auto-run simulation", value=bool(sim_state.get("running")))
    if auto and not sim_state.get("running"):
        api_post("/api/v1/sim/start")
    elif not auto and sim_state.get("running"):
        api_post("/api/v1/sim/stop")

    col1, col2 = st.columns(2)
    if col1.button("⏩ Tick", width="stretch"):
        api_post("/api/v1/sim/tick?n=5")
        st.rerun()
    if col2.button("🔄 Refresh", width="stretch"):
        st.rerun()

    speed = st.slider("Tick interval (s)", 1, 10, 2)
    if auto:
        st.caption(f"Sim clock advances ~{60 / speed * 5:.0f} min/min")

# ---------------------------------------------------------------- pages
if page == "Live Lot":
    st.header("Live Lot Map")
    data = api_get("/api/v1/slots")
    if not data:
        st.warning("Waiting for API…")
        st.stop()

    summary = data["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Occupancy", f"{summary['occupancy_pct']}%",
              f"{summary['occupied']}/{summary['total']} slots")
    c2.metric("Free", summary["free"])
    stats = api_get("/api/v1/stats", {})
    c3.metric("Revenue (sim)", f"₹{stats.get('revenue', 0):,.0f}")
    c4.metric("Rejections (sim)", stats.get("sim_rejections", 0))

    st.divider()
    # per-section availability at a glance
    sections = api_get("/api/v1/sections", {}).get("sections", [])
    if sections:
        st.subheader("Parking sections")
        sdf = pd.DataFrame([
            {"Section": f"{s['icon']} {s['name']}", "Free": s["free"],
             "Occupied": s["occupied"], "Total": s["total"],
             "Occupancy %": s["occupancy_pct"], "From ₹/h": s["price_from"]}
            for s in sections])
        st.dataframe(sdf, width="stretch", hide_index=True)
        st.caption(" · ".join(f"{s['icon']} {s['entrance']}" for s in sections))

    zones = {}
    for s in data["slots"]:
        zones.setdefault(s["zone"], []).append(s)

    for sec, slots in [(k, [s for s in data["slots"] if s.get("section") == k])
                       for k in SECTION_META]:
        if not slots:
            continue
        meta = SECTION_META[sec]
        occ = sum(1 for s in slots if s["status"] == "occupied")
        st.markdown(
            f"### {meta['icon']} {meta['name']} "
            f"<span style='color:#94a3b8;font-size:0.8em'>"
            f"({len(slots) - occ} free · {occ}/{len(slots)} occupied)</span>",
            unsafe_allow_html=True)
        for zone in sorted({s["zone"] for s in slots}):
            zslots = zones.get(zone, [])
            if not zslots:
                continue
            ztype = zslots[0]["slot_type"]
            zocc = sum(1 for s in zslots if s["status"] == "occupied")
            icon = ZONE_ICONS.get(ztype, "")
            st.markdown(f"**Zone {zone}** {icon} "
                        f"<span style='color:#94a3b8;font-size:0.85em'>({ztype} · "
                        f"{zocc}/{len(zslots)} occupied · ₹{zslots[0]['price_per_hour']}/h)"
                        f"</span>", unsafe_allow_html=True)
            cards_html = "<div class='lotmap'>"
            for s in zslots:
                color = STATUS_COLORS.get(s["status"], "#94a3b8")
                cards_html += (
                    f"<div class='slotcard' style='background:{color}' "
                    f"title='{s['code']} · {s['status']}'>"
                    f"{s['code']}</div>")
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.caption("🟩 free · 🟥 occupied · 🟧 reserved · ⬜ maintenance")

elif page == "Analytics":
    st.header("Analytics")
    hist = api_get("/api/v1/occupancy/history?limit=400")
    stats = api_get("/api/v1/stats", {})

    if not hist or not hist["history"]:
        st.info("No history yet — run the simulation for a few ticks.")
        st.stop()

    df = pd.DataFrame(hist["history"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Current occupancy", f"{stats.get('occupancy_pct', 0)}%")
    c2.metric("Active bookings", stats.get("active_bookings", 0))
    c3.metric("Completed bookings", stats.get("completed_bookings", 0))

    fig = px.line(df, x="sim_hour", y="occupied", title="Occupied slots vs sim hour",
                  markers=True)
    fig.add_hline(y=df["total"].iloc[-1], line_dash="dot",
                  annotation_text="capacity")
    st.plotly_chart(fig, width="stretch")

    fig2 = px.area(df, x="sim_hour", y="revenue", title="Cumulative revenue (₹)")
    st.plotly_chart(fig2, width="stretch")

    if stats.get("by_zone"):
        zdf = pd.DataFrame([
            {"zone": z, "occupancy_pct": v["occupancy_pct"],
             "total": v["total"], "occupied": v["occupied"]}
            for z, v in stats["by_zone"].items()])
        fig3 = px.bar(zdf, x="zone", y="occupancy_pct", color="zone",
                      title="Occupancy by zone (%)", text="occupancy_pct")
        st.plotly_chart(fig3, width="stretch")

    rej = df["rejections"].diff().fillna(0)
    fig4 = px.bar(x=df["sim_hour"], y=rej, title="Rejections per tick")
    fig4.update_xaxes(title="sim hour")
    fig4.update_yaxes(title="rejections")
    st.plotly_chart(fig4, width="stretch")

elif page == "Book Parking":
    st.header("Book Parking")
    users = api_get("/api/v1/users", {}).get("users", [])
    if not users:
        st.info("No users yet (seeded users appear after `python scripts/init_db.py`)")
        user_id = 1
    else:
        user_map = {f"{u['name']} · {u['vehicle_plate']} ({u['vehicle_type']})":
                    u for u in users}
        sel = st.selectbox("User", list(user_map))
        user = user_map[sel]
        user_id = user["id"]

    c1, c2 = st.columns(2)
    vtype = c1.selectbox("Vehicle type", ["car", "ev", "disabled", "bike"],
                         index=["car", "ev", "disabled", "bike"].index(
                             user.get("vehicle_type", "car")) if users else 0)
    stay = c2.slider("Stay duration (minutes)", 15, 480, 60, step=15)

    if st.button("🔮 Get AI recommendation", type="primary"):
        rec = api_post("/api/v1/recommend",
                       {"vehicle_type": vtype, "stay_minutes": stay})
        if rec.get("_error"):
            st.error(rec["_error"])
        else:
            mode = rec["assigned_by"]
            badge = "🤖 RL agent" if mode == "rl" else "⚙️ heuristic"
            st.success(f"Slot **{rec['slot']['code']}** (Zone {rec['slot']['zone']}) "
                       f"· {badge} · est. ₹{rec['estimated_price']}")
            if st.button("✅ Confirm check-in", key="confirm"):
                chk = api_post("/api/v1/checkin",
                               {"user_id": user_id, "vehicle_type": vtype,
                                "stay_minutes": stay})
                if chk.get("_error"):
                    st.error(chk["_error"])
                else:
                    st.balloons()
                    st.success(chk["message"])

    st.divider()
    slots = api_get("/api/v1/slots", {}).get("slots", [])
    occ = sum(1 for s in slots if s["status"] == "occupied")
    st.caption(f"{occ} slots currently occupied")
    bookings = api_get("/api/v1/bookings/active", {}).get("bookings", [])
    if bookings:
        st.dataframe(pd.DataFrame(bookings), width="stretch",
                     hide_index=True)
    else:
        st.caption("No active sessions.")

elif page == "Admin":
    st.header("Admin")
    st.subheader("Retrain RL agent")
    steps = st.number_input("Timesteps", 10_000, 500_000, 60_000, step=10_000)
    if st.button("🏋️ Start training", type="primary"):
        with st.spinner("Training PPO agent… (a few minutes)"):
            result = api_post("/api/v1/admin/train?steps=" + str(int(steps)))
        if result.get("_error"):
            st.error(result["_error"])
        else:
            st.success(result.get("message", "done"))

    st.divider()
    st.subheader("Reset simulation")
    if st.button("♻️ Reset lot (release all slots)"):
        api_post("/api/v1/admin/reset")
        st.success("Lot reset")

    st.divider()
    st.subheader("Raw API state")
    st.json(api_get("/api/v1/sim/state", {}) or {})
