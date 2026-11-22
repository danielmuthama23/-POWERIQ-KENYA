"""Kenya problem intelligence: additive analysis layer on top of the existing service.

Everything here is computed from the synthetic world, the trained models and the dispatch engine already in the app.
The only external numbers are the Kenya reference statistics taken from the user-supplied briefing (KENYA_BRIEF); they are
shown as "reported in the briefing, not independently verified" and are never used to calibrate the synthetic data.
Techniques: Monte Carlo loss-of-load (binomial unit outages driven by the maintenance model), STL decomposition,
KMeans plant clustering, dispatch/LP what-if studies, Pareto inspection planning, rule-based findings + LLM/local narrative.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from .service import LTWP_CORRIDOR_LIMIT_MW, PowerIQ

# Figures quoted in the user-supplied briefing (source it cites: EPRA statistics / press reports, year ended June 2026).
KENYA_BRIEF = {
    "source": "User-supplied briefing document (cites EPRA 2026 statistics and press reports). Not independently verified here.",
    "system_loss_pct": 21.38, "allowable_loss_pct": 16.5,
    "avg_interruption_hours": 3.23,
    "reserve_margin_pct": 3.34, "recommended_reserve_pct": [20, 35],
    "curtailment_gwh": {"geothermal": 392.36, "wind": 11.58, "total": 403.94},
}
MODULES = [  # the briefing's proposed modules -> where each lives in this build
    {"module": "GenerationIQ", "status": "implemented", "where": "Generation Intelligence / Renewable Intelligence (LightGBM forecasts)"},
    {"module": "GridIQ", "status": "implemented", "where": "Grid Intelligence (DC power flow, N-1 screening)"},
    {"module": "PredictIQ", "status": "implemented", "where": "Predictive Maintenance (XGBoost) + Plant Fleet clustering"},
    {"module": "StorageIQ", "status": "implemented", "where": "Storage Optimization (LP) + Curtailment-to-Storage study"},
    {"module": "DemandIQ", "status": "implemented", "where": "Overview/Generation (LightGBM) + Demand Structure (STL)"},
    {"module": "TheftIQ", "status": "implemented", "where": "Energy Loss Detection + Theft Inspection Planner"},
    {"module": "ClimateIQ", "status": "partial", "where": "Climate Intelligence + Drought/Wind-lull stress tests (synthetic weather; no live feeds)"},
    {"module": "Digital Twin", "status": "implemented", "where": "National Digital Twin (scenario engine)"},
    {"module": "MaintenanceIQ", "status": "partial", "where": "Maintenance risk ranking; no work-order scheduling optimiser"},
    {"module": "Energy Analytics", "status": "implemented", "where": "Kenya Problems + Insights pages"},
]
NOT_IMPLEMENTED = ["PyTorch/TensorFlow LSTM or Temporal Fusion Transformer forecasting", "Kafka/Spark streaming", "PostgreSQL/TimescaleDB/PostGIS",
                   "MQTT/OPC UA ingestion", "YOLO/OpenCV drone inspection", "Cesium globe", "Power BI/Grafana", "Azure deployment (script untested)"]

_cache: dict[tuple[int, str], Any] = {}
_lock = threading.Lock()


def _cached(s: PowerIQ, key: str, fn):
    k = (id(s), key)
    with _lock:
        if k in _cache:
            return _cache[k]
    v = fn()
    with _lock:
        _cache[k] = v
    return v


def _r(x, n=1):
    return None if x is None else round(float(x), n)


# ------------------------------------------------------------------ 1. scorecard
def scorecard(s: PowerIQ) -> dict:
    def build():
        w, n = s.world, s.world.n_hist
        lv = s.losses_view()
        cv = s.curtailment_view()
        b = s.baseline
        ev = w.curtailment_events
        ev30 = ev[ev.time >= w.times[n - 1] - pd.Timedelta(days=30)]
        by_src = ev30.groupby("source").curtailed_mw.sum()
        tot = float(by_src.sum()) or 1.0
        hist = w.system.iloc[n - 24 * 30:n]
        rows = [
            {"id": "losses", "problem": "System losses", "brief": f"{KENYA_BRIEF['system_loss_pct']}% vs {KENYA_BRIEF['allowable_loss_pct']}% allowable",
             "synthetic": f"{lv['totals']['loss_pct']:.1f}% (latest month, synthetic)", "model": "Loss detector (IsolationForest + rule)", "page": "losses"},
            {"id": "reserve", "problem": "Thin reserve margin", "brief": f"{KENYA_BRIEF['reserve_margin_pct']}% vs 20-35% recommended",
             "synthetic": f"{float(np.min(b['reserve_margin_pct'])):.1f}% min / {float(np.mean(b['reserve_margin_pct'])):.1f}% mean over next 72 h",
             "model": "Dispatch engine + Monte Carlo LOLP", "page": "reserve"},
            {"id": "curtailment", "problem": "Curtailed geothermal/wind", "brief": f"{KENYA_BRIEF['curtailment_gwh']['total']} GWh/yr (97.1% geothermal)",
             "synthetic": f"{cv['total_curtailed_mwh']/1000:.2f} GWh in 30 d; {by_src.get('geothermal', 0)/tot*100:.0f}% geothermal",
             "model": "Dispatch engine + battery LP", "page": "curtailment"},
            {"id": "interruptions", "problem": "Supply interruptions", "brief": f"{KENYA_BRIEF['avg_interruption_hours']} h average interruption",
             "synthetic": f"{int((hist.unserved_mw > 0.5).sum())} unserved hours in last 30 d; {int((b['unserved'] > 0.5).sum())} forecast in next 72 h",
             "model": "Dispatch engine", "page": "reserve"},
            {"id": "drought", "problem": "Drought / hydro dependence", "brief": "Hydro output falls in droughts",
             "synthetic": f"Current hydro water factor {float(w.system.water_factor.iloc[n - 1]):.2f}", "model": "Drought stress test", "page": "climate"},
            {"id": "wind", "problem": "Wind & solar intermittency", "brief": "Output fluctuates; evening peak shortfall",
             "synthetic": f"Evening-peak unserved forecast {float(b['unserved'][:24].sum()):.0f} MWh (24 h)", "model": "Forecasts + stress tests", "page": "climate"},
            {"id": "equipment", "problem": "Equipment breakdowns", "brief": "Turbines/transformers fail unexpectedly",
             "synthetic": f"{int(s.maint_risk.flagged.sum())} of {len(s.maint_risk)} units flagged at risk", "model": "XGBoost maintenance", "page": "plants"},
            {"id": "theft", "problem": "Illegal connections / theft", "brief": "Meter tampering and illegal connections",
             "synthetic": f"{int(lv['totals']['ntl_mwh']):,} MWh estimated non-technical loss (latest month)", "model": "Loss detector", "page": "theft"},
        ]
        return {"data_origin": "synthetic vs briefing", "brief": KENYA_BRIEF, "rows": rows, "modules": MODULES, "not_implemented": NOT_IMPLEMENTED,
                "note": "Synthetic values are NOT calibrated to the briefing's figures; the comparison shows where each Kenyan problem is modelled, not that the data matches reality."}
    return _cached(s, "scorecard", build)


# ------------------------------------------------------------------ 2. reserve risk (Monte Carlo LOLP)
def reserve_risk(s: PowerIQ, n_sims: int = 2000, seed: int = 7) -> dict:
    def build():
        H = 72
        rng = np.random.default_rng(seed)
        b = s.baseline
        req = b["required"][:H]
        pot = s.plant_pot0[:H].copy()
        ids = s.plants.plant_id.values
        i_ltwp = int(np.where(ids == "LTWP")[0][0])
        pot[:, i_ltwp] = np.minimum(pot[:, i_ltwp], LTWP_CORRIDOR_LIMIT_MW)
        # unit outage probability over 72 h from the maintenance model (assumption: 3/14 of its 14-day fault probability)
        mr = s.maint_risk.groupby("plant_id").risk_probability.mean()
        p = np.array([min(0.5, float(mr.get(pid, 0.0)) * 3 / 14) for pid in ids])
        nu = s.plants.n_units.values.astype(int)
        failed = rng.binomial(nu[None, :], p[None, :], size=(n_sims, len(ids)))
        mult = 1 - failed / np.maximum(nu[None, :], 1)                     # [N,P] capacity fraction retained
        sig = 0.03
        if s.fb.demand_rel_err_q:
            k = min(s.fb.demand_rel_err_q, key=lambda x: abs(x - 24))
            q10, q90 = s.fb.demand_rel_err_q[k]
            sig = max(0.01, (abs(q90) + abs(q10)) / 2.563)
        z = 0.7 * rng.normal(size=(n_sims, 1)) + 0.7 * rng.normal(size=(n_sims, H))
        req_s = req[None, :] * (1 + sig * z)
        sup = (pot[None, :, :] * mult[:, None, :]).sum(axis=2)             # [N,H]
        short = np.clip(req_s - sup, 0, None)
        eens = float(short.sum(axis=1).mean())
        lolp_h = (short > 1.0).mean(axis=0)
        contrib = []
        for j, pid in enumerate(ids):
            if p[j] <= 0:
                continue
            sup_j = sup + pot[None, :, j] * (1 - mult[:, j])[:, None]
            e_j = float(np.clip(req_s - sup_j, 0, None).sum(axis=1).mean())
            contrib.append({"plant_id": pid, "plant": s.plants.name.values[j], "source": s.plants.source.values[j],
                            "outage_prob_72h": _r(p[j], 3), "eens_reduction_mwh": _r(eens - e_j, 1)})
        contrib.sort(key=lambda r: -r["eens_reduction_mwh"])
        tot_short = short.sum(axis=1)
        margin = (sup.mean(axis=0) - req) / np.maximum(req, 1) * 100
        return {"data_origin": "synthetic+model", "method": "Monte Carlo (binomial unit outages from the XGBoost maintenance risk, correlated demand error)",
                "n_sims": n_sims, "demand_error_sigma": _r(sig, 4), "horizon_hours": H,
                "p_any_shortfall_72h": _r((tot_short > 1.0).mean(), 3), "expected_unserved_mwh_72h": _r(eens, 1),
                "p95_unserved_mwh_72h": _r(np.percentile(tot_short, 95), 1), "worst_hour_lolp": _r(lolp_h.max(), 3),
                "hourly": [{"h": h, "required_mw": _r(req[h], 0), "expected_supply_mw": _r(sup[:, h].mean(), 0), "lolp": _r(lolp_h[h], 3),
                            "p10_margin_pct": _r((np.percentile(sup[:, h], 10) - req[h]) / max(req[h], 1) * 100, 1)} for h in range(H)],
                "mean_margin_pct": _r(margin.mean(), 1), "min_margin_pct": _r(margin.min(), 1),
                "top_risk_contributors": contrib[:8], "briefing_reserve_margin_pct": KENYA_BRIEF["reserve_margin_pct"],
                "assumptions": ["Plant outage lasts the whole 72 h window (conservative)", "72 h outage probability = 3/14 of the 14-day model probability",
                                "Wind/solar/hydro at forecast potential; no storage; transmission limits ignored except the Lake Turkana corridor"]}
    return _cached(s, "reserve", build)


# ------------------------------------------------------------------ 3. curtailment -> storage
def curtailment_storage(s: PowerIQ) -> dict:
    def build():
        cv = s.curtailment_view()
        base = s.baseline
        w, n = s.world, s.world.n_hist
        ev = w.curtailment_events
        ev30 = ev[ev.time >= w.times[n - 1] - pd.Timedelta(days=30)]
        src = ev30.groupby("source").curtailed_mw.sum()
        reason = ev30.groupby("reason").curtailed_mw.sum()

        def stat(r):
            return {"curtailed_mwh": _r(r["curtailed"].sum()), "thermal_mwh": _r(r["thermal"].sum()), "unserved_mwh": _r(r["unserved"].sum()),
                    "peak_unserved_mw": _r(r["unserved"].max()),
                    "corridor_limited_mwh": _r(r["corridor_curtailed"].sum())}
        b0 = stat(base)
        opts = []
        for mw, mwh in [(50, 200), (100, 400), (200, 800), (300, 1200)]:
            r = s.run_dispatch(battery={"power_mw": mw, "energy_mwh": mwh})
            if not r["battery"] or not r["battery"].get("feasible"):
                continue
            st = stat(r)
            opts.append({"power_mw": mw, "energy_mwh": mwh, **st,
                         "curtailment_recovered_mwh": _r(b0["curtailed_mwh"] - st["curtailed_mwh"]),
                         "thermal_avoided_mwh": _r(b0["thermal_mwh"] - st["thermal_mwh"]),
                         "unserved_avoided_mwh": _r(b0["unserved_mwh"] - st["unserved_mwh"])})
        return {"data_origin": "synthetic+model", "window_days": 30, "curtailed_mwh_30d": _r(cv["total_curtailed_mwh"]),
                "by_source_mwh": {k: _r(v) for k, v in src.items()}, "by_reason_mwh": {k: _r(v) for k, v in reason.items()},
                "by_hour_of_day_mwh": cv["by_hour_of_day_mwh"], "by_plant": sorted(cv["by_plant"], key=lambda r: -r["mwh"])[:10],
                "next_72h_baseline": b0, "battery_options": opts,
                "briefing_annual_curtailment_gwh": KENYA_BRIEF["curtailment_gwh"],
                "insight": "Storage near load centres cannot recover energy curtailed upstream by the Lake Turkana corridor limit (corridor_limited_mwh); that needs transmission, not batteries.",
                "method": "Each option re-runs the 72 h dispatch with the LP battery (charging free from curtailment). Sizes are illustrative, not investment advice.",
                "caveat": "Synthetic curtailment is produced by the toy merit-order model; it is not a reproduction of Kenya's reported curtailment."}
    return _cached(s, "curt", build)


# ------------------------------------------------------------------ 4. climate stress tests
def climate_stress(s: PowerIQ) -> dict:
    def build():
        src = s.plants.source.values
        scen = [("Baseline", {}), ("Mild drought (hydro -20%)", {"hydro": .8}), ("Drought (hydro -40%)", {"hydro": .6}),
                ("Severe drought (hydro -60%)", {"hydro": .4}), ("Wind lull (wind -50%)", {"wind": .5}),
                ("Dry + calm (hydro -40%, wind -50%)", {"hydro": .6, "wind": .5}), ("Cloudy week (solar -50%)", {"solar": .5})]
        out = []
        for name, f in scen:
            pot = s.plant_pot0.copy()
            for k, v in f.items():
                pot[:, src == k] *= v
            r = s.run_dispatch(plant_pot=pot)
            out.append({"scenario": name, "thermal_mwh": _r(r["thermal"].sum(), 0), "unserved_mwh": _r(r["unserved"].sum(), 0),
                        "hours_unserved": int((r["unserved"] > 0.5).sum()), "min_reserve_pct": _r(r["reserve_margin_pct"].min()),
                        "peak_thermal_mw": _r(r["thermal"].max(), 0), "curtailed_mwh": _r(r["curtailed"].sum(), 0)})
        base = out[0]
        for o in out:
            o["extra_thermal_mwh"] = _r(o["thermal_mwh"] - base["thermal_mwh"], 0)
            o["extra_unserved_mwh"] = _r(o["unserved_mwh"] - base["unserved_mwh"], 0)
        wf = s.world.system.water_factor.iloc[s.world.n_hist - 60 * 24:s.world.n_hist:24].values.astype(float)
        slope = float(np.polyfit(np.arange(len(wf)), wf, 1)[0] * 30)
        return {"data_origin": "synthetic+model", "horizon_hours": 72, "scenarios": out,
                "water_factor": {"current": _r(wf[-1], 3), "trend_per_30d": _r(slope, 3), "min_60d": _r(wf.min(), 3), "max_60d": _r(wf.max(), 3)},
                "note": "Stress tests scale forecast potential and re-dispatch. Weather and hydrology are synthetic; no rainfall/satellite feeds are connected."}
    return _cached(s, "climate", build)


# ------------------------------------------------------------------ 5. theft inspection planner
def theft_planner(s: PowerIQ, top: int = 15) -> dict:
    def build():
        lt = s.lb.latest
        fl = lt[lt.flagged == 1].sort_values("ntl_mwh_est", ascending=False)
        tot = float(lt.ntl_mwh_est.sum()) or 1.0
        cust = s.world.feeders.set_index("feeder_id").n_customers
        cum = fl.ntl_mwh_est.cumsum() / tot
        pareto = [{"feeders_inspected": k, "share_of_estimated_ntl_pct": _r(float(cum.iloc[min(k, len(cum)) - 1]) * 100, 1)}
                  for k in (5, 10, 25, 50) if len(cum)]
        regs = s.losses_view()["regions"]
        allow = KENYA_BRIEF["allowable_loss_pct"]
        gap = [{"region": r["region"], "loss_pct": _r(r["loss_pct"]), "ntl_pct": _r(r["ntl_pct"]), "tech_pct": _r(r["tech_pct"]),
                "mwh_above_allowable": _r(max(0.0, (r["loss_pct"] - allow) / 100 * r["input_mwh"]), 0)} for r in regs]
        gap.sort(key=lambda r: -r["mwh_above_allowable"])
        rows = [{"rank": i + 1, "feeder_id": r.feeder_id, "region": r.region, "cell_id": r.cell_id, "ntl_est_pct": _r(r.ntl_est_pct, 2),
                 "ntl_mwh_est": _r(r.ntl_mwh_est, 1), "customers": int(cust.get(r.feeder_id, 0)),
                 "mwh_per_100_customers": _r(r.ntl_mwh_est / max(cust.get(r.feeder_id, 1), 1) * 100, 2)} for i, r in enumerate(fl.head(top).itertuples())]
        return {"data_origin": "synthetic+model", "flagged_feeders": int(len(fl)), "total_ntl_mwh": _r(tot, 0), "inspection_list": rows,
                "pareto": pareto, "regions_vs_allowable": gap, "allowable_loss_pct": allow,
                "method": "Feeders ranked by estimated non-technical energy (flagged by the loss detector); Pareto shows recovery if the top-k are inspected.",
                "caveat": "Estimates come from synthetic data with injected theft; a flag is a lead for field verification, never proof of theft."}
    return _cached(s, "theft", build)


# ------------------------------------------------------------------ 6. plant fleet profile + clustering
def plant_fleet(s: PowerIQ) -> dict:
    def build():
        w, n = s.world, s.world.n_hist
        P = s.plants
        gen = w.plant_gen.iloc[n - 720:n]
        av = w.plant_avail.iloc[n - 720:n]
        ev = w.curtailment_events
        ev30 = ev[ev.time >= w.times[n - 1] - pd.Timedelta(days=30)].groupby("plant_id").curtailed_mw.sum()
        mr = s.maint_risk
        rmax = mr.groupby("plant_id").risk_probability.max()
        nfl = mr.groupby("plant_id").flagged.sum()
        disp = s.baseline["plant_disp"].sum(axis=0)
        tot_d = float(disp.sum()) or 1.0
        rows = []
        for j, r in enumerate(P.itertuples()):
            cap = float(r.capacity_mw)
            g = gen[r.plant_id]
            rows.append({"plant_id": r.plant_id, "name": r.name, "source": r.source, "region": r.region, "capacity_mw": cap,
                         "capacity_factor_pct": _r(g.mean() / cap * 100), "availability_pct": _r(av[r.plant_id].mean() * 100),
                         "curtailed_mwh_30d": _r(ev30.get(r.plant_id, 0.0)), "curtailment_pct_of_potential": _r(ev30.get(r.plant_id, 0.0) / max(g.sum() + ev30.get(r.plant_id, 0.0), 1) * 100, 2),
                         "max_unit_risk": _r(rmax.get(r.plant_id, 0.0), 3), "flagged_units": int(nfl.get(r.plant_id, 0)),
                         "forecast_72h_mwh": _r(disp[j], 0), "share_of_72h_supply_pct": _r(disp[j] / tot_d * 100, 1)})
        df = pd.DataFrame(rows)
        feats = df[["capacity_factor_pct", "availability_pct", "curtailment_pct_of_potential", "max_unit_risk"]].fillna(0).values
        Z = StandardScaler().fit_transform(feats)
        km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Z)
        names = ["capacity factor", "availability", "curtailment", "maintenance risk"]
        clusters = []
        for c in range(4):
            cen = km.cluster_centers_[c]
            k = int(np.argmax(np.abs(cen)))
            word = {(0, 1): "high utilisation", (0, 0): "low utilisation", (1, 1): "high availability", (1, 0): "low availability",
                    (2, 1): "curtailment-exposed", (2, 0): "low curtailment", (3, 1): "elevated maintenance risk", (3, 0): "low maintenance risk"}[(k, int(cen[k] > 0))]
            members = df.index[km.labels_ == c]
            clusters.append({"cluster": c, "label": word, "dominant_feature": names[k], "n_plants": int(len(members)),
                             "capacity_mw": _r(df.capacity_mw[members].sum(), 0), "plants": df.name[members].tolist()})
        df["cluster"] = km.labels_
        return {"data_origin": "synthetic+model", "plants": df.to_dict("records"), "clusters": clusters,
                "method": "KMeans (k=4) on standardised capacity factor, availability, curtailment share and max unit risk. Cluster labels name the feature furthest from the fleet mean.",
                "note": "Plant names are real; capacities, outputs and risks are synthetic."}
    return _cached(s, "fleet", build)


# ------------------------------------------------------------------ 7. demand structure (STL)
def demand_structure(s: PowerIQ) -> dict:
    def build():
        w, n = s.world, s.world.n_hist
        y = pd.Series(w.system.total_demand_mw.iloc[n - 24 * 56:n].values, index=w.times[n - 24 * 56:n])
        res = STL(y, period=24, robust=True).fit()
        var = lambda x: float(np.var(x))
        fs = max(0.0, 1 - var(res.resid) / var(res.seasonal + res.resid))
        ft = max(0.0, 1 - var(res.resid) / var(res.trend + res.resid))
        daily = y.resample("D").mean()
        prof = y.groupby(y.index.hour).mean()
        dow = y.groupby(y.index.dayofweek).mean()
        tr = res.trend.resample("D").mean()
        peak = float(y.max())
        ev = y[(y.index.hour >= 18) & (y.index.hour <= 21)].sum() / y.sum() * 100
        return {"data_origin": "synthetic", "window_days": 56, "method": "STL (period 24 h, robust) via statsmodels",
                "seasonal_strength": _r(fs, 3), "trend_strength": _r(ft, 3), "load_factor_pct": _r(y.mean() / peak * 100, 1),
                "peak_mw": _r(peak, 0), "min_mw": _r(y.min(), 0), "peak_to_valley": _r(peak / max(float(y.min()), 1), 2),
                "evening_peak_share_pct": _r(ev, 1), "peak_hour_of_day": int(prof.idxmax()),
                "hourly_profile_mw": [_r(prof.get(h, 0), 0) for h in range(24)],
                "day_of_week_mw": [_r(dow.get(d, 0), 0) for d in range(7)],
                "trend_daily": [{"date": str(d.date()), "trend_mw": _r(v, 0), "mean_mw": _r(daily.loc[d], 0)} for d, v in tr.items()],
                "trend_change_pct_56d": _r((tr.iloc[-1] / tr.iloc[0] - 1) * 100, 2),
                "residual_sd_mw": _r(float(np.std(res.resid)), 1)}
    return _cached(s, "demand", build)


# ------------------------------------------------------------------ 8. AI analysis (findings + narrative)
def findings(s: PowerIQ) -> list[dict]:
    rr, cs, cl, th, pf, ds = reserve_risk(s), curtailment_storage(s), climate_stress(s), theft_planner(s), plant_fleet(s), demand_structure(s)
    f: list[dict] = []
    if rr["p_any_shortfall_72h"] >= 0.05 or rr["min_margin_pct"] < 10:
        f.append({"id": "F-RESERVE", "severity": "high" if rr["p_any_shortfall_72h"] >= 0.25 else "medium", "area": "Reserve margin",
                  "finding": f"Monte Carlo gives a {rr['p_any_shortfall_72h']*100:.0f}% chance of at least one shortfall hour in the next 72 h "
                             f"(expected unserved {rr['expected_unserved_mwh_72h']:.0f} MWh, P95 {rr['p95_unserved_mwh_72h']:.0f} MWh; minimum expected margin {rr['min_margin_pct']:.1f}%).",
                  "evidence": {"top_contributors": rr["top_risk_contributors"][:3]},
                  "action": "Review planned maintenance timing for the top contributing plants and confirm standby thermal/import availability for the evening peak."})
    best = max(cs["battery_options"], key=lambda o: o["curtailment_recovered_mwh"] + o["unserved_avoided_mwh"], default=None)
    if cs["curtailed_mwh_30d"] > 0 and best:
        f.append({"id": "F-CURTAIL", "severity": "medium", "area": "Curtailment & storage",
                  "finding": f"{cs['curtailed_mwh_30d']:.0f} MWh was curtailed in the last 30 d (synthetic). A {best['power_mw']} MW / {best['energy_mwh']} MWh battery recovers "
                             f"{best['curtailment_recovered_mwh']:.0f} MWh and avoids {best['thermal_avoided_mwh']:.0f} MWh thermal in the next 72 h.",
                  "evidence": {"options": cs["battery_options"]}, "action": "Run the Storage Optimization page with your own price and site data before sizing any procurement."})
    d40, d60 = next(x for x in cl["scenarios"] if x["scenario"].startswith("Drought (")), next(x for x in cl["scenarios"] if x["scenario"].startswith("Severe"))
    if d40["extra_unserved_mwh"] > 0 or d40["extra_thermal_mwh"] > 0:
        f.append({"id": "F-DROUGHT", "severity": "high" if d40["extra_unserved_mwh"] > 0 else "medium", "area": "Drought exposure",
                  "finding": f"A 40% hydro shortfall adds {d40['extra_thermal_mwh']:.0f} MWh thermal and {d40['extra_unserved_mwh']:.0f} MWh unserved energy over 72 h; "
                             f"a 60% shortfall adds {d60['extra_thermal_mwh']:.0f} MWh thermal and {d60['extra_unserved_mwh']:.0f} MWh unserved.",
                  "evidence": {"scenarios": cl["scenarios"]}, "action": "Pre-position thermal fuel and import contracts ahead of dry-season forecasts; track reservoir levels (no live feed in this build)."})
    over = [r for r in th["regions_vs_allowable"] if r["mwh_above_allowable"] > 0]
    if over:
        top = th["pareto"][1] if len(th["pareto"]) > 1 else th["pareto"][0]
        f.append({"id": "F-LOSS", "severity": "medium", "area": "Losses & theft",
                  "finding": f"{len(over)} regions exceed the {th['allowable_loss_pct']}% allowable loss; {th['flagged_feeders']} feeders flagged. Inspecting the top {top['feeders_inspected']} "
                             f"covers {top['share_of_estimated_ntl_pct']}% of estimated non-technical loss.",
                  "evidence": {"regions": over[:5], "pareto": th["pareto"]}, "action": "Schedule field verification for the ranked feeder list; treat flags as leads, not accusations."})
    risky = [p for p in pf["plants"] if p["flagged_units"] > 0]
    if risky:
        cap = sum(p["capacity_mw"] * p["flagged_units"] / max(1, next(q.n_units for q in s.plants.itertuples() if q.plant_id == p["plant_id"])) for p in risky)
        f.append({"id": "F-MAINT", "severity": "medium", "area": "Power station maintenance",
                  "finding": f"{sum(p['flagged_units'] for p in risky)} units across {len(risky)} stations are flagged (about {cap:.0f} MW of capacity exposed to a 14-day fault).",
                  "evidence": {"plants": [{k: p[k] for k in ("name", "flagged_units", "max_unit_risk", "capacity_mw")} for p in sorted(risky, key=lambda p: -p["max_unit_risk"])[:5]]},
                  "action": "Prioritise inspection of the highest-risk units; precision is modest (see Metrics), so confirm with condition monitoring."})
    if ds["evening_peak_share_pct"] > 20:
        f.append({"id": "F-PEAK", "severity": "low", "area": "Demand structure",
                  "finding": f"18:00-21:59 carries {ds['evening_peak_share_pct']}% of energy; peak {ds['peak_mw']:.0f} MW vs valley {ds['min_mw']:.0f} MW (ratio {ds['peak_to_valley']}), seasonal strength {ds['seasonal_strength']}.",
                  "evidence": {"profile": ds["hourly_profile_mw"]}, "action": "Evening-peak storage and demand response are the natural levers; evaluate in the Digital Twin."})
    order = {"high": 0, "medium": 1, "low": 2}
    f.sort(key=lambda x: order[x["severity"]])
    return f


def ai_analysis(s: PowerIQ, question: str = "Summarise the main reliability and efficiency problems in Kenya's power system and what to do first.") -> dict:
    from . import copilot
    fs = findings(s)
    rr, cs, cl = reserve_risk(s), curtailment_storage(s), climate_stress(s)
    secs = {
        "measured_facts": [f"{r['problem']}: synthetic {r['synthetic']}" for r in scorecard(s)["rows"][:5]],
        "model_predictions": [x["finding"] for x in fs if x["id"] in ("F-RESERVE", "F-DROUGHT", "F-CURTAIL")] or ["No elevated prediction-based risks found."],
        "assumptions": rr["assumptions"][:2] + ["All data is synthetic; briefing statistics are shown for context only and are not used for calibration."],
        "recommendations": [f"[{x['severity']}] {x['area']}: {x['action']}" for x in fs],
    }
    evidence = {"findings": fs, "scorecard": scorecard(s)["rows"], "briefing": KENYA_BRIEF}
    text, err = copilot._llm_narrative(evidence, question)
    mode = "openai" if text else "local-template"
    if not text:
        head = (f"{len(fs)} findings ranked by severity: " + "; ".join(f"{x['area']} ({x['severity']})" for x in fs)) if fs else "No findings above thresholds."
        text = copilot._local_narrative(secs, head)
    return {"data_origin": "synthetic+model", "mode": mode, "llm_status": "ok" if mode == "openai" else f"not used: {err}", "question": question,
            "narrative": text, "findings": fs, "sections": secs,
            "notice": "Findings are rule-based over measured model outputs; the narrative (LLM or template) only restates them. Advisory only; human approval required."}
