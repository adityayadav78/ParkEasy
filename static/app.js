// ParkEasy live dashboard — polls the FastAPI backend and renders the lot.
const API = "";

const STATUS_COLORS = {
  free: "#22c55e",
  occupied: "#ef4444",
  reserved: "#f59e0b",
  maintenance: "#94a3b8",
};
const ZONE_TYPES = {
  A: "standard", B: "standard", P: "premium", E: "ev", D: "disabled", M: "bike",
};

async function jget(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
async function jpost(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

function fmtHour(h) {
  const hh = Math.floor(h) % 24;
  const mm = Math.round((h - Math.floor(h)) * 60);
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

async function refreshHealth() {
  try {
    const h = await jget("/api/v1/health");
    setStatus("api-status", "API online", "ok");
    document.getElementById("rl-status").textContent =
      "RL: " + (h.rl_model_loaded ? "PPO agent ✅" : "heuristic");
    document.getElementById("sim-clock").textContent =
      "sim " + fmtHour(h.sim_hour);
  } catch {
    setStatus("api-status", "API offline", "bad");
  }
}

function setStatus(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = "pill" + (cls ? " " + cls : "");
}

async function refreshLot() {
  let data;
  try {
    data = await jget("/api/v1/slots");
  } catch {
    return;
  }
  const s = data.summary;
  document.getElementById("kpi-occ").textContent = s.occupancy_pct + "%";
  document.getElementById("kpi-free").textContent = s.free;

  try {
    const st = await jget("/api/v1/stats");
    document.getElementById("kpi-rev").textContent = "₹" + Math.round(st.revenue).toLocaleString();
    document.getElementById("kpi-rej").textContent = st.sim_rejections;
    document.getElementById("sim-clock").textContent =
      "sim " + fmtHour(st.sim_hour);
  } catch { /* non-fatal */ }

  const zones = {};
  for (const slot of data.slots) (zones[slot.zone] ||= []).push(slot);

  const lot = document.getElementById("lot");
  lot.innerHTML = "";
  for (const [zone, slots] of Object.entries(zones)) {
    const occ = slots.filter((x) => x.status === "occupied").length;
    const block = document.createElement("div");
    block.className = "zone-block";
    block.innerHTML = `
      <div class="zone-title"><b>Zone ${zone}</b> · ${ZONE_TYPES[zone] || ""} ·
        ${occ}/${slots.length} occupied · ₹${slots[0].price_per_hour}/h</div>
      <div class="lotmap"></div>`;
    const map = block.querySelector(".lotmap");
    for (const slot of slots) {
      const card = document.createElement("div");
      card.className = "slotcard";
      card.style.background = STATUS_COLORS[slot.status] || "#94a3b8";
      card.textContent = slot.code;
      card.title = `${slot.code} · ${slot.status}`;
      map.appendChild(card);
    }
    lot.appendChild(block);
  }
}

async function tick5() {
  try { await jpost("/api/v1/sim/tick?n=5"); } catch (e) { console.error(e); }
  refreshAll();
}
async function startSim() {
  try { await jpost("/api/v1/sim/start", { interval: 2 }); } catch (e) { console.error(e); }
  refreshAll();
}
async function stopSim() {
  try { await jpost("/api/v1/sim/stop"); } catch (e) { console.error(e); }
  refreshAll();
}

async function recommendAndCheckin() {
  const out = document.getElementById("book-result");
  out.className = "";
  out.textContent = "Asking the RL agent…";
  const vehicle_type = document.getElementById("veh-type").value;
  const stay_minutes = parseInt(document.getElementById("stay").value, 10) || 60;
  try {
    const rec = await jpost("/api/v1/recommend", { vehicle_type, stay_minutes });
    const badge = rec.assigned_by === "rl" ? "🤖 RL agent" : "⚙️ heuristic";
    out.className = "ok";
    out.textContent = `Slot ${rec.slot.code} (${badge}, est ₹${rec.estimated_price}) — checking in…`;
    const users = await jget("/api/v1/users");
    const uid = users.users.length ? users.users[0].id : 1;
    const chk = await jpost("/api/v1/checkin", {
      user_id: uid, vehicle_type, stay_minutes,
    });
    out.className = "ok";
    out.textContent = chk.message;
    refreshAll();
  } catch (e) {
    out.className = "err";
    out.textContent = "⚠️ " + e.message;
  }
}

function refreshAll() {
  refreshHealth();
  refreshLot();
}

document.getElementById("btn-tick").addEventListener("click", tick5);
document.getElementById("btn-start").addEventListener("click", startSim);
document.getElementById("btn-stop").addEventListener("click", stopSim);
document.getElementById("btn-rec").addEventListener("click", recommendAndCheckin);

refreshAll();
setInterval(refreshAll, 3000);
