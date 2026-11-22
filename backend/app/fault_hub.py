"""Fault Hub: locate and prioritise malfunctions of power stations, substations and transmission lines.

Alert sources, all located by coordinates:
  * telemetry anomalies (IsolationForest detections on transformer/turbine sensors)   -> status "active"
  * plant availability drops / derates from the generation record                    -> "active"
  * the grid-incident log (line trips, breaker failures, transformer faults...)      -> "active" / "resolved"
  * model predictions: units with high 14-day failure risk, forecast line loading, N-1 single points of failure -> "predicted"
  * operator / SCADA reports posted to POST /faults/report                           -> "reported" (genuinely real time)
The "live" clock replays the synthetic world (default: its latest observed hour); only reported faults are real-time.
Nothing here controls equipment: dispatch requests go to the human approval queue.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from . import geo
from .security import get_audit
from .service import PowerIQ

SEV_W = {"critical": 4, "high": 3, "medium": 2, "low": 1}
SLA_H = {"critical": 1, "high": 4, "medium": 24, "low": 72}   # illustrative response targets, not a utility policy
KENYA_BBOX = (-5.2, 5.6, 33.5, 42.5)                           # lat_min, lat_max, lon_min, lon_max
INCIDENT = {
    "line_trip": ("high", "Line tripped: patrol the corridor, review protection records and confirm the fault location before any reclose."),
    "overload": ("medium", "Reduce loading through redispatch or a topology review; verify thermal ratings."),
    "breaker_failure": ("high", "Isolate the bay, confirm backup protection and send a substation crew."),
    "protection_misoperation": ("medium", "Review relay settings and event records; test the protection scheme."),
    "transformer_fault": ("critical", "Isolate the transformer, run dissolved-gas and oil checks, arrange load transfer."),
    "lightning": ("low", "Inspect insulators and surge arresters on the affected section."),
}
_lock = threading.Lock()
_static: dict[int, dict] = {}
_tables = False


# ------------------------------------------------------------------ geometry and static assets
def hav_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def static(s: PowerIQ) -> dict:
    with _lock:
        if id(s) in _static:
            return _static[id(s)]
    w = s.world
    subs = {r.bus_id: {"id": r.bus_id, "name": r.name, "kv": int(r.kv), "lat": float(r.lat), "lon": float(r.lon)} for r in w.substations.itertuples()}
    plants = {r.plant_id: {"id": r.plant_id, "name": r.name, "source": r.source, "capacity_mw": float(r.capacity_mw), "lat": float(r.lat), "lon": float(r.lon),
                           "region": r.region, "bus": r.bus_id, "n_units": int(r.n_units)} for r in s.plants.itertuples()}
    lines = {}
    for r in s.net.lines.itertuples():
        a, b = subs[r.from_bus], subs[r.to_bus]
        lines[r.line_id] = {"id": r.line_id, "name": r.line_id.replace("-", " to "), "from": r.from_bus, "to": r.to_bus, "kv": int(r.kv), "capacity_mw": float(r.capacity_mw),
                            "coords": [[a["lon"], a["lat"]], [b["lon"], b["lat"]]], "lat": (a["lat"] + b["lat"]) / 2, "lon": (a["lon"] + b["lon"]) / 2}
    cells = s.cells
    bus_mw = (pd.Series(s.baseline["cell_demand"][:24].mean(axis=0), index=cells.cell_id.values).groupby(cells.bus_id.values).sum()).to_dict()
    # illustrative mapping of telemetry assets to sites (the synthetic world has no site field for them)
    trf_sites = sorted(subs.values(), key=lambda x: (-x["kv"], x["id"]))
    tur_sites = sorted([p for p in plants.values() if p["source"] in ("hydro", "geothermal")], key=lambda p: -p["capacity_mw"])
    units = {r.unit_id: float(r.rated_mw) for r in w.units.itertuples()}
    out = {"subs": subs, "plants": plants, "lines": lines, "bus_mw": {k: float(v) for k, v in bus_mw.items()}, "trf_sites": trf_sites, "tur_sites": tur_sites, "unit_mw": units}
    with _lock:
        _static[id(s)] = out
    return out


def locate(s: PowerIQ, lat: float, lon: float) -> dict:
    """Nearest region/cell and nearest named asset for any coordinate."""
    c = s.cells
    d = ((c.lat.values - lat) ** 2 + ((c.lon.values - lon) * math.cos(math.radians(lat))) ** 2)
    i = int(np.argmin(d))
    st = static(s)
    best = None
    for kind, group in (("substation", st["subs"]), ("plant", st["plants"]), ("line", st["lines"])):
        for a in group.values():
            if kind == "line":
                dist = min(hav_km(lat, lon, a["coords"][0][1] + (a["coords"][1][1] - a["coords"][0][1]) * t, a["coords"][0][0] + (a["coords"][1][0] - a["coords"][0][0]) * t) for t in np.linspace(0, 1, 21))
            else:
                dist = hav_km(lat, lon, a["lat"], a["lon"])
            if best is None or dist < best[0]:
                best = (dist, kind, a)
    return {"region": str(c.region.iloc[i]), "cell_id": str(c.cell_id.iloc[i]), "bus": str(c.bus_id.iloc[i]),
            "nearest_asset": {"type": best[1], "id": best[2]["id"], "name": best[2]["name"], "km": round(best[0], 1)}}


def assets_view(s: PowerIQ) -> dict:
    st = static(s)
    return {"data_origin": "synthetic", "plants": list(st["plants"].values()), "substations": list(st["subs"].values()), "lines": list(st["lines"].values()),
            "outline": [list(p) for p in geo.KENYA_OUTLINE], "cells": [{"id": r.cell_id, "poly": [list(p) for p in r.polygon]} for r in s.cells.itertuples()],
            "note": "Illustrative network and outline; real station names, synthetic capacities and coordinates."}


# ------------------------------------------------------------------ storage (reports + acknowledgements)
def _tables_ready():
    global _tables
    a = get_audit()
    if _tables:
        return a
    with a.lock:
        a.db.execute("""CREATE TABLE IF NOT EXISTS fault_reports (id TEXT PRIMARY KEY, created TEXT, reporter TEXT, asset_type TEXT, asset_id TEXT, name TEXT,
            lat REAL, lon REAL, severity TEXT, description TEXT, status TEXT, updated TEXT, created_epoch REAL)""")
        a.db.execute("""CREATE TABLE IF NOT EXISTS fault_community (id TEXT PRIMARY KEY, created TEXT, reporter TEXT, kind TEXT, lat REAL, lon REAL, description TEXT,
            status TEXT, updated TEXT, created_epoch REAL)""")
        a.db.execute("""CREATE TABLE IF NOT EXISTS fault_detections (id TEXT PRIMARY KEY, created TEXT, source TEXT, camera_id TEXT, label TEXT, obj TEXT, confidence REAL,
            lat REAL, lon REAL, ts TEXT, image_ref TEXT, model_version TEXT, status TEXT, created_epoch REAL)""")
        a.db.execute("CREATE TABLE IF NOT EXISTS fault_ack (alert_id TEXT PRIMARY KEY, by_ TEXT, ts TEXT, note TEXT)")
        a.db.commit()
    _tables = True
    return a


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 3 * 3600)) + "+03:00"


def reset_for_tests():
    global _tables
    _tables = False
    with _lock:
        _static.clear()


# ------------------------------------------------------------------ alert construction
def _mk(s: PowerIQ, *, id: str, kind: str, asset_type: str, asset_id: str, name: str, lat: float, lon: float, severity: str, status: str, title: str,
        detail: str, action: str, source: str, mw: float, started: str | None = None, hours_until: float | None = None, geometry: list | None = None, extra: dict | None = None) -> dict:
    loc = locate(s, lat, lon)
    return {"id": id, "kind": kind, "asset_type": asset_type, "asset_id": asset_id, "name": name, "lat": round(lat, 4), "lon": round(lon, 4), "geometry": geometry,
            "severity": severity, "status": status, "title": title, "detail": detail, "recommended_action": action, "source": source, "mw_at_risk": round(float(mw), 1),
            "started": started, "hours_until": hours_until, "region": loc["region"], "cell_id": loc["cell_id"], "nearest_asset": loc["nearest_asset"],
            "respond_within_h": SLA_H[severity], **(extra or {})}


def _priority(a: dict) -> float:
    w = {"active": 1.0, "reported": 1.0, "detected": 0.95, "acknowledged": 0.9, "unverified": 0.55, "resolved": 0.15}.get(a["status"])
    if w is None:   # predicted
        w = 0.7 * max(0.3, 1 - (a.get("hours_until") or 0) / 24)
    return round(SEV_W[a["severity"]] * (1 + math.log1p(a["mw_at_risk"]) / math.log1p(300)) * w, 3)


def _sev_from_mw(mw: float) -> str:
    return "critical" if mw >= 60 else "high" if mw >= 25 else "medium"


def live(s: PowerIQ, offset_h: int = 0, include_predicted: bool = True, include_resolved: bool = False) -> dict:
    w, n = s.world, s.world.n_hist
    offset_h = int(np.clip(offset_h, -72, 71))
    t_idx = n - 1 + offset_h
    T = w.times[t_idx]
    st = static(s)
    alerts: list[dict] = []

    # 1. telemetry anomalies (model detections in the last 6 h of the clock)
    an = s.anomalies
    an = an[(an.is_anomaly == 1) & (an.time <= T) & (an.time > T - pd.Timedelta(hours=6))]
    for aid, g in an.groupby("asset_id"):
        k = int(aid.split("-")[1]) - 1
        if aid.startswith("TRF"):
            site = st["trf_sites"][k % len(st["trf_sites"])]
            at, mw, nm = "substation", st["bus_mw"].get(site["id"], 0.0), f"{site['name']} substation ({aid})"
            action = "Inspect the transformer: oil temperature/pressure and loading; prepare load transfer if readings keep rising."
            aid_ = site["id"]
        else:
            site = st["tur_sites"][k % len(st["tur_sites"])]
            at, mw, nm = "plant", site["capacity_mw"] / max(site["n_units"], 1), f"{site['name']} turbine ({aid})"
            action = "Inspect the turbine: vibration and bearing temperature; consider derating until checked."
            aid_ = site["id"]
        last = g.iloc[-1]
        persistent = len(g) >= 3
        sev = "critical" if persistent and mw >= 60 else "high" if persistent else "medium"
        alerts.append(_mk(s, id=f"anom:{aid}:{str(g.time.min())[:13]}", kind="telemetry_anomaly", asset_type=at, asset_id=aid_, name=nm, lat=site["lat"], lon=site["lon"], severity=sev,
                          status="active", title="Abnormal sensor behaviour", detail=f"{len(g)} anomalous readings in 6 h; strongest on the {last.driver_channel} channel (score {last.score:.2f}).",
                          action=action, source="Model: IsolationForest on telemetry", mw=mw, started=str(g.time.min())))

    # 2. plant availability drop (derate or forced outage)
    av = w.plant_avail.iloc[t_idx]
    for pid, a in av.items():
        if a < 0.98:
            p = st["plants"][pid]
            lost = (1 - a) * p["capacity_mw"]
            run = 0
            while t_idx - run >= 0 and w.plant_avail.iloc[t_idx - run][pid] < 0.98 and run < 24 * 14:
                run += 1
            sev = _sev_from_mw(lost)
            alerts.append(_mk(s, id=f"out:{pid}:{str(w.times[t_idx - run + 1])[:13]}", kind="availability_loss", asset_type="plant", asset_id=pid, name=p["name"], lat=p["lat"], lon=p["lon"],
                              severity=sev, status="active", title=f"{(1 - a) * 100:.0f}% of capacity unavailable",
                              detail=f"Availability {a * 100:.0f}% for {run} h, about {lost:.0f} MW lost.", action="Confirm the cause with the plant operator, expected return time, and cover the gap in dispatch.",
                              source="Generation record (availability)", mw=lost, started=str(w.times[t_idx - run + 1])))

    # 3. incident log (active now, or resolved within 24 h)
    inc = w.grid_incidents
    for r in inc.itertuples():
        end = r.time + pd.Timedelta(hours=float(r.duration_h))
        if r.time <= T and (end >= T or (include_resolved and end >= T - pd.Timedelta(hours=24))):
            sev0, action = INCIDENT.get(r.kind, ("medium", "Investigate."))
            tgt = r.target
            if tgt in st["lines"]:
                ln = st["lines"][tgt]
                at, lat, lon, geom, nm, mw = "line", ln["lat"], ln["lon"], ln["coords"], ln["name"] + " line", ln["capacity_mw"] * 0.4
            elif tgt in st["subs"]:
                sb = st["subs"][tgt]
                at, lat, lon, geom, nm, mw = "substation", sb["lat"], sb["lon"], None, sb["name"] + " substation", st["bus_mw"].get(tgt, 0.0)
            else:
                continue
            resolved = end < T
            alerts.append(_mk(s, id=f"inc:{tgt}:{str(r.time)[:13]}", kind=r.kind, asset_type=at, asset_id=tgt, name=nm, lat=lat, lon=lon, geometry=geom,
                              severity=sev0, status="resolved" if resolved else "active", title=r.kind.replace("_", " ").capitalize(),
                              detail=f"Started {str(r.time)[:16]}, duration {float(r.duration_h):.1f} h" + (" (cleared)." if resolved else " (ongoing)."), action=action,
                              source="Incident log", mw=mw, started=str(r.time)))

    # 4. predictions
    f0 = max(offset_h, 0)
    banners = []
    if include_predicted:
        mr = s.maint_risk
        for r in mr[mr.risk_probability >= 0.6].itertuples():
            p = st["plants"][r.plant_id]
            mw = st["unit_mw"].get(r.unit_id, 0.0)
            alerts.append(_mk(s, id=f"risk:{r.unit_id}", kind="predicted_failure", asset_type="plant", asset_id=r.plant_id, name=f"{p['name']} unit {r.unit_id}", lat=p["lat"], lon=p["lon"],
                              severity="high" if r.risk_probability >= 0.8 else "medium", status="predicted", title=f"{r.risk_probability * 100:.0f}% fault probability within 14 days",
                              detail=f"Drivers: {'; '.join(r.drivers) if isinstance(r.drivers, (list, tuple)) else r.drivers}.", action="Schedule an inspection before the risk window ends; consider derating if sensors keep trending up.",
                              source="Model: XGBoost maintenance risk", mw=mw, hours_until=24.0))
        load = s.baseline["loading"][f0:f0 + 12]
        for j, lid in enumerate(s.net.lines.line_id.values):
            mx = float(load[:, j].max())
            if mx >= 0.70:
                ln = st["lines"][lid]
                hr = int(load[:, j].argmax())
                alerts.append(_mk(s, id=f"load:{lid}:{f0}", kind="predicted_overload", asset_type="line", asset_id=lid, name=ln["name"] + " line", lat=ln["lat"], lon=ln["lon"], geometry=ln["coords"],
                                  severity="high" if mx >= 0.9 else "medium", status="predicted", title=f"Loading forecast to reach {mx * 100:.0f}%",
                                  detail=f"Peak in about {hr + 1} h (DC power-flow approximation).", action="Review redispatch or switching options before the peak; confirm thermal rating.",
                                  source="Forecast: DC power flow", mw=mx * ln["capacity_mw"], hours_until=float(hr + 1)))
        for c in s.n1_screen()["contingencies_with_violations"] if False else []:
            pass
        n1 = s.n1_screen()
        for v in n1["violations"]:
            lid = v["outage"]
            if lid not in st["lines"]:
                continue
            ln = st["lines"][lid]
            over = bool(v.get("overloads"))
            mw = min(st["bus_mw"].get(ln["to"], 0.0), st["bus_mw"].get(ln["from"], 0.0)) if v.get("islanding") else ln["capacity_mw"] * 0.3
            alerts.append(_mk(s, id=f"n1:{lid}", kind="n1_vulnerability", asset_type="line", asset_id=lid, name=ln["name"] + " line", lat=ln["lat"], lon=ln["lon"], geometry=ln["coords"],
                              severity="medium" if over else "low", status="predicted", title="Single point of failure (N-1)",
                              detail=("If this line trips, " + ("part of the network is islanded" if v.get("islanding") else "other lines overload") + ("; overloads: " + ", ".join(map(str, v["overloads"])) if over else "") + "."),
                              action="Consider a second path, remedial switching plan, or protection review.", source="Screening: N-1 contingency analysis", mw=mw, hours_until=24.0))
        un = s.baseline["unserved"][f0:f0 + 24]
        if (un > 0.5).any():
            h = int(np.argmax(un > 0.5))
            banners.append({"level": "high", "text": f"System-level shortfall forecast: {float(un.sum()):.0f} MWh unserved in the next 24 h, first around +{h + 1} h."})
        rm = float(np.min(s.baseline["reserve_margin_pct"][f0:f0 + 24]))
        if rm < 10:
            banners.append({"level": "medium", "text": f"Minimum forecast reserve margin {rm:.1f}% in the next 24 h."})

    # 5. real-time operator / SCADA reports
    a = _tables_ready()
    with a.lock:
        rep = a.db.execute("SELECT id,created,reporter,asset_type,asset_id,name,lat,lon,severity,description,status,updated FROM fault_reports WHERE status!='resolved' ORDER BY created_epoch DESC LIMIT 100").fetchall()
        res = a.db.execute("SELECT id,created,reporter,asset_type,asset_id,name,lat,lon,severity,description,status,updated FROM fault_reports WHERE status='resolved' ORDER BY created_epoch DESC LIMIT 20").fetchall() if include_resolved else []
    for r in rep + res:
        rid, created, reporter, atype, aid, nm, lat, lon, sev, desc, stt, _ = r
        geom = st["lines"][aid]["coords"] if atype == "line" and aid in st["lines"] else None
        mw = (st["plants"][aid]["capacity_mw"] if atype == "plant" and aid in st["plants"] else st["bus_mw"].get(aid, 0.0) if atype == "substation" else
              st["lines"][aid]["capacity_mw"] * 0.4 if atype == "line" and aid in st["lines"] else 0.0)
        if mw == 0.0:
            loc = locate(s, lat, lon)
            mw = st["bus_mw"].get(loc["bus"], 0.0) * 0.3
        alerts.append(_mk(s, id=f"rep:{rid}", kind="operator_report", asset_type=atype, asset_id=aid or "", name=nm, lat=lat, lon=lon, geometry=geom, severity=sev,
                          status="resolved" if stt == "resolved" else "reported", title="Reported fault", detail=desc, action="Dispatch a crew to the coordinates, confirm the fault and isolate if needed.",
                          source=f"Report by {reporter} at {created}", mw=mw, started=created, extra={"report_id": rid}))

    # 6. model detections (cameras / satellite) and 7. community concerns
    alerts.extend(_detection_alerts(s, st))
    alerts.extend(_community_alerts(s, st))

    # acknowledgements, priority, ordering
    with a.lock:
        acks = {r[0]: {"by": r[1], "ts": r[2], "note": r[3]} for r in a.db.execute("SELECT alert_id,by_,ts,note FROM fault_ack").fetchall()}
    for x in alerts:
        x["ack"] = acks.get(x["id"])
        if x["ack"] and x["status"] in ("active", "reported", "detected"):
            x["status"] = "acknowledged"
        x["priority"] = _priority(x)
    alerts.sort(key=lambda x: -x["priority"])
    for i, x in enumerate(alerts):
        x["rank"] = i + 1
    live_set = [x for x in alerts if x["status"] in ("active", "reported", "detected", "acknowledged")]
    summary = {"total": len(alerts), "active_now": len(live_set), "predicted": sum(x["status"] == "predicted" for x in alerts), "unverified": sum(x["status"] == "unverified" for x in alerts), "detected": sum(x["status"] == "detected" for x in alerts), "resolved": sum(x["status"] == "resolved" for x in alerts),
               "by_severity": {k: sum(x["severity"] == k for x in alerts if x["status"] != "resolved") for k in SEV_W},
               "by_type": {k: sum(x["asset_type"] == k for x in alerts if x["status"] != "resolved") for k in ("plant", "substation", "line", "other")},
               "mw_at_risk_active": round(sum(x["mw_at_risk"] for x in live_set), 1), "reported": sum(x["status"] in ("reported",) or x["kind"] == "operator_report" for x in alerts)}
    return {"data_origin": "synthetic+operator reports", "as_of": str(T), "offset_h": offset_h, "now_observed": str(w.times[n - 1]), "server_time": _now_iso(), "alerts": alerts,
            "summary": summary, "banners": banners,
            "notice": ("Simulated live feed: detections and outages replay the synthetic world; only fault reports posted to this hub are genuinely real time. "
                       "Response targets are illustrative. POWERIQ never operates equipment; dispatch requests need human approval.")}


# ------------------------------------------------------------------ reports, acknowledgement, dispatch
class FaultError(ValueError):
    pass


def add_report(s: PowerIQ, reporter: str, asset_type: str, asset_id: str | None, lat: float | None, lon: float | None, severity: str, description: str, name: str | None) -> dict:
    st = static(s)
    if severity not in SEV_W:
        raise FaultError("severity must be critical, high, medium or low")
    if asset_type not in ("plant", "substation", "line", "other"):
        raise FaultError("asset_type must be plant, substation, line or other")
    nm = name
    if asset_id:
        grp = {"plant": st["plants"], "substation": st["subs"], "line": st["lines"]}.get(asset_type, {})
        a = grp.get(asset_id)
        if a is None:
            raise FaultError(f"Unknown {asset_type} id '{asset_id}'")
        lat, lon, nm = a["lat"], a["lon"], nm or a["name"]
    if lat is None or lon is None:
        raise FaultError("Give an asset id or coordinates (lat and lon).")
    if not (KENYA_BBOX[0] <= lat <= KENYA_BBOX[1] and KENYA_BBOX[2] <= lon <= KENYA_BBOX[3]):
        raise FaultError("Coordinates are outside Kenya's bounding box.")
    loc = locate(s, lat, lon)
    nm = nm or f"Fault near {loc['nearest_asset']['name']}"
    rid = "FR-" + uuid.uuid4().hex[:8].upper()
    a_ = _tables_ready()
    with a_.lock:
        a_.db.execute("INSERT INTO fault_reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (rid, _now_iso(), reporter, asset_type, asset_id or "", nm[:80], float(lat), float(lon), severity, description.strip()[:300], "open", _now_iso(), time.time()))
        a_.db.commit()
    return {"id": rid, "alert_id": f"rep:{rid}", "name": nm, "lat": lat, "lon": lon, **loc}


def resolve_report(rid: str) -> bool:
    a = _tables_ready()
    with a.lock:
        cur = a.db.execute("UPDATE fault_reports SET status='resolved', updated=? WHERE id=? AND status!='resolved'", (_now_iso(), rid))
        a.db.commit()
        return cur.rowcount > 0


def acknowledge(alert_id: str, by: str, note: str | None) -> None:
    a = _tables_ready()
    with a.lock:
        a.db.execute("INSERT OR REPLACE INTO fault_ack VALUES (?,?,?,?)", (alert_id, by, _now_iso(), (note or "")[:200]))
        a.db.commit()


def dispatch_request(s: PowerIQ, alert_id: str, by: str, offset_h: int = 0) -> dict | None:
    al = next((x for x in live(s, offset_h)["alerts"] if x["id"] == alert_id), None)
    if al is None:
        return None
    text = (f"Dispatch a field crew to {al['name']} ({al['lat']:.3f}, {al['lon']:.3f}), {al['region']}: {al['title']}. Priority #{al['rank']}, respond within "
            f"{al['respond_within_h']} h. Action: {al['recommended_action']}")
    rid = get_audit().add_recommendation("fault-hub", text, by, {"alert_id": alert_id, "severity": al["severity"], "lat": al["lat"], "lon": al["lon"]})
    return {"recommendation_id": rid, "text": text, "status": "pending"}


# ------------------------------------------------------------------ model detections and community concerns
CONDITION = {"smoke": "smoking", "fire": "on fire", "sparking": "sparking", "missing_component": "missing", "damaged": "damaged", "sagging_line": "sagging",
             "tree_on_line": "obstructed by vegetation", "pole_down": "down", "vandalism": "vandalised or stolen", "other": "faulty"}
DET_SEV = {"smoke": "critical", "fire": "critical", "sparking": "critical", "missing_component": "high", "damaged": "high", "pole_down": "high", "vandalism": "high",
           "sagging_line": "medium", "tree_on_line": "medium", "other": "medium"}
DEFAULT_OBJ = {"plant": "Plant equipment", "substation": "Transformer", "line": "Power line", "other": "Equipment"}
DET_LABELS = tuple(CONDITION)
COMM_KINDS = ("smoke", "fire", "sparking", "missing_component", "damaged", "pole_down", "vandalism", "sagging_line", "tree_on_line", "other")
HIGH_CONF, MIN_CONF = 0.80, 0.50
_rate: dict[str, list[float]] = {}


def _near_site(s: PowerIQ, lat: float, lon: float, obj: str | None = None, max_km: float = 5.0) -> dict:
    """Asset within max_km (else 'other') with its id, type and MW, plus the place wording. A transformer prefers a substation, a turbine a plant."""
    st = static(s)
    loc = locate(s, lat, lon)
    prefer = None
    o = (obj or "").lower()
    if any(k in o for k in ("transformer", "substation", "switchgear", "breaker")):
        prefer = "substation"
    elif any(k in o for k in ("turbine", "generator", "plant", "penstock")):
        prefer = "plant"
    elif any(k in o for k in ("pole", "conductor", "line", "insulator", "tower", "wire")):
        prefer = "line"
    na = loc["nearest_asset"]
    if prefer and prefer != na["type"]:
        grp = {"substation": st["subs"], "plant": st["plants"]}.get(prefer)
        if grp:
            b = min(grp.values(), key=lambda x: hav_km(lat, lon, x["lat"], x["lon"]))
            d = hav_km(lat, lon, b["lat"], b["lon"])
            if d <= max_km * 3:
                na = {"type": prefer, "id": b["id"], "name": b["name"], "km": round(d, 1)}
    atype, aid, mw, nm = "other", "", 0.0, None
    if na["km"] <= max_km * (3 if prefer == na["type"] else 1):
        atype, aid, nm = na["type"], na["id"], na["name"]
        if atype == "substation":
            mw = st["bus_mw"].get(aid, 0.0)
        elif atype == "plant":
            mw = st["plants"][aid]["capacity_mw"]
        elif atype == "line":
            mw = st["lines"][aid]["capacity_mw"] * 0.4
    region = loc["region"].replace("_", " ").title()
    suffix = {"substation": " substation", "plant": " power station", "line": " line"}.get(atype, "")
    place = f"{region} (near {nm}{suffix})" if nm else region
    return {"asset_type": atype, "asset_id": aid, "mw": mw, "place": place, "region": loc["region"], "loc": loc}


def describe(s: PowerIQ, label: str, lat: float, lon: float, obj: str | None = None) -> dict:
    """Plain-language headline and dispatch instruction, e.g. 'Transformer smoking at Nairobi (near EMBAKASI)'."""
    site = _near_site(s, lat, lon, obj)
    o = (obj or DEFAULT_OBJ[site["asset_type"]]).strip().capitalize()
    cond = CONDITION.get(label, "faulty")
    headline = f"{o} {cond} at {site['place']}"
    action = f"Dispatch a crew to the coordinates ({lat:.4f}, {lon:.4f}). " + (
        "Keep people clear, inform the fire service if there is smoke or flame, and isolate the equipment only on the control room's instruction."
        if label in ("smoke", "fire", "sparking") else "Inspect, confirm the fault on site and report back; isolate only on the control room's instruction.")
    return {"headline": headline, "action": action, "site": site, "severity": DET_SEV.get(label, "medium")}


def _check_coords(lat: float, lon: float):
    if not (KENYA_BBOX[0] <= lat <= KENYA_BBOX[1] and KENYA_BBOX[2] <= lon <= KENYA_BBOX[3]):
        raise FaultError("Coordinates are outside Kenya's bounding box.")


def add_detections(s: PowerIQ, by: str, items: list[dict]) -> dict:
    """Ingest detections from an externally trained model (camera feeds, satellite tiles). Stored as given; confidence decides the status."""
    a = _tables_ready()
    out, skipped = [], []
    with a.lock:
        for d in items:
            try:
                if d["label"] not in DET_LABELS:
                    raise FaultError(f"unknown label '{d['label']}'")
                _check_coords(d["lat"], d["lon"])
            except FaultError as e:
                skipped.append({"detection_id": d.get("detection_id"), "reason": str(e)})
                continue
            did = d.get("detection_id") or "DET-" + uuid.uuid4().hex[:8].upper()
            conf = float(d["confidence"])
            status = "detected" if conf >= HIGH_CONF else "unverified" if conf >= MIN_CONF else "logged"
            a.db.execute("INSERT OR REPLACE INTO fault_detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (did, _now_iso(), d["source"], (d.get("camera_id") or "")[:60], d["label"], (d.get("object") or "")[:40], conf, float(d["lat"]), float(d["lon"]),
                          (d.get("observed_at") or "")[:30], (d.get("image_ref") or "")[:200], (d.get("model_version") or "")[:40], status, time.time()))
            out.append({"detection_id": did, "status": status, "alert_id": f"det:{did}" if status != "logged" else None})
        a.db.commit()
    return {"accepted": len(out), "skipped": skipped, "results": out,
            "rules": f"confidence >= {HIGH_CONF} becomes a detected alert, {MIN_CONF} to {HIGH_CONF} goes to the review queue, below {MIN_CONF} is only logged."}


def _detection_alerts(s: PowerIQ, st: dict) -> list[dict]:
    a = _tables_ready()
    with a.lock:
        rows = a.db.execute("SELECT id,source,camera_id,label,obj,confidence,lat,lon,ts,image_ref,model_version,status,created FROM fault_detections WHERE status IN ('detected','unverified','confirmed') "
                            "ORDER BY created_epoch DESC LIMIT 200").fetchall()
    res = []
    for did, src, cam, label, obj, conf, lat, lon, ts, img, mv, stt, created in rows:
        ds = describe(s, label, lat, lon, obj or None)
        site = ds["site"]
        sev = ds["severity"] if conf >= HIGH_CONF else ("high" if ds["severity"] == "critical" else ds["severity"])
        status = "detected" if stt in ("detected", "confirmed") else "unverified"
        res.append(_mk(s, id=f"det:{did}", kind="model_detection", asset_type=site["asset_type"], asset_id=site["asset_id"], name=ds["headline"], lat=lat, lon=lon,
                       geometry=st["lines"][site["asset_id"]]["coords"] if site["asset_type"] == "line" else None, severity=sev, status=status,
                       title=f"{label.replace('_', ' ').capitalize()} detected ({conf * 100:.0f}% confidence)",
                       detail=f"Seen by {src}{' ' + cam if cam else ''}" + (f" at {ts}" if ts else "") + (f", image {img}" if img else "") + (f", model {mv}" if mv else "") +
                              (". Needs a human to confirm." if status == "unverified" else "."),
                       action=ds["action"], source=f"Trained model ({src})", mw=site["mw"], started=ts or created, extra={"detection_id": did, "confidence": conf, "label": label}))
    return res


def set_detection_status(did: str, status: str) -> bool:
    a = _tables_ready()
    with a.lock:
        cur = a.db.execute("UPDATE fault_detections SET status=? WHERE id=? AND status IN ('detected','unverified','confirmed')", (status, did))
        a.db.commit()
        return cur.rowcount > 0


def add_community(s: PowerIQ, reporter: str, kind: str, lat: float, lon: float, description: str) -> dict:
    if kind not in COMM_KINDS:
        raise FaultError("Unknown concern type.")
    _check_coords(lat, lon)
    now = time.time()
    with _lock:
        recent = [x for x in _rate.get(reporter, []) if now - x < 3600]
        if len(recent) >= 10:
            raise FaultError("Too many community reports from this key in the last hour (limit 10).")
        _rate[reporter] = recent + [now]
    cid = "CR-" + uuid.uuid4().hex[:8].upper()
    a = _tables_ready()
    with a.lock:
        a.db.execute("INSERT INTO fault_community VALUES (?,?,?,?,?,?,?,?,?,?)", (cid, _now_iso(), reporter, kind, float(lat), float(lon), description.strip()[:300], "unverified", _now_iso(), now))
        a.db.commit()
    ds = describe(s, kind, lat, lon)
    return {"id": cid, "alert_id": f"com:{cid}", "status": "unverified", "headline": ds["headline"], "place": ds["site"]["place"],
            "message": "Thank you. Your report is in the queue and will be checked by an operator before any crew is sent. If anyone is in danger, call the emergency services now."}


def _community_alerts(s: PowerIQ, st: dict) -> list[dict]:
    a = _tables_ready()
    with a.lock:
        rows = a.db.execute("SELECT id,created,kind,lat,lon,description,status FROM fault_community WHERE status IN ('unverified','verified') ORDER BY created_epoch DESC LIMIT 200").fetchall()
    res = []
    for cid, created, kind, lat, lon, desc, stt in rows:
        near = sum(1 for r in rows if r[0] != cid and r[2] == kind and hav_km(lat, lon, r[3], r[4]) <= 1.0)   # independent reports of the same thing nearby
        ds = describe(s, kind, lat, lon)
        site = ds["site"]
        sev = ds["severity"] if stt == "verified" or near >= 1 else ("high" if ds["severity"] == "critical" else ds["severity"])
        status = "reported" if stt == "verified" else "unverified"
        res.append(_mk(s, id=f"com:{cid}", kind="community_report", asset_type=site["asset_type"], asset_id=site["asset_id"], name=ds["headline"], lat=lat, lon=lon,
                       geometry=st["lines"][site["asset_id"]]["coords"] if site["asset_type"] == "line" else None, severity=sev, status=status,
                       title="Community concern" + (" (verified)" if stt == "verified" else f" ({near + 1} similar reports within 1 km)" if near else " (unverified)"),
                       detail=desc or kind.replace("_", " "), action=ds["action"] if stt == "verified" or near else "Verify first: check nearby cameras or call the reporter's area office, then dispatch if confirmed.",
                       source="Community report", mw=site["mw"], started=created, extra={"community_id": cid, "corroborating_reports": near}))
    return res


def set_community_status(cid: str, status: str) -> bool:
    a = _tables_ready()
    with a.lock:
        cur = a.db.execute("UPDATE fault_community SET status=?, updated=? WHERE id=? AND status IN ('unverified','verified')", (status, _now_iso(), cid))
        a.db.commit()
        return cur.rowcount > 0
