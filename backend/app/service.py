"""POWERIQ analytics service: owns the synthetic world, trained models, forecasts, dispatch/scenario engine and map layers."""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import geo, ml
from .network import DCNetwork
from .optimize import BatterySpec, optimize_battery
from .synthetic import HORIZON_MAX, World, make_world

TLOSS = 0.025
LTWP_CORRIDOR_LIMIT_MW = 250.0
SCENARIO_LABEL = ("Illustrative merit-order dispatch with a DC-approximation network. NOT a validated AC power-flow "
                  "study; do not use for operational decisions.")


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S+03:00")


class PowerIQ:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.world: World = make_world(seed)
        w = self.world
        self.fb = ml.train_forecasts(w, seed)
        self.mb = ml.train_maintenance(w, seed)
        self.ab = ml.train_anomaly(w, seed)
        self.lb = ml.train_loss(w, seed)
        self.net = DCNetwork(w.substations, w.lines)
        self.trained_at = dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).isoformat(timespec="seconds")
        self.cells = w.cells.reset_index(drop=True)
        self.plants = w.plants.reset_index(drop=True)
        self._prepare_forecasts()
        self.maint_risk = ml.maintenance_risk(w, self.mb)
        self.anomalies = ml.recent_anomalies(w, self.ab)
        self.baseline = self.run_dispatch()
        self.model_versions = {"forecast": self.fb.version, "maintenance": self.mb.version,
                               "anomaly": self.ab.version, "loss": self.lb.version}

    # ------------------------------------------------------------------ forecasts
    def _prepare_forecasts(self) -> None:
        w, n = self.world, self.world.n_hist
        self.dem_fc = ml.forecast_demand(w, self.fb, HORIZON_MAX, self.seed)
        g = ml.forecast_generation(w, self.fb, HORIZON_MAX, self.seed)
        self.gen_fc, self.gen_lo, self.gen_hi = g["mean"], g["lo"], g["hi"]
        self.times_fc = self.dem_fc.index
        H = HORIZON_MAX
        sh = self.cells.set_index("cell_id").share
        reg = self.cells.region.values
        self.cell_dem = np.column_stack([self.dem_fc[r].values * s for r, s in zip(reg, self.cells.share.values)])
        self.cap = self.plants.capacity_mw.values.astype(float)
        self.avail_fc = w.plant_avail.values[n:n + H]                      # scheduled availability (assumption: known)
        wf = float(w.system.water_factor.iloc[n - 1])
        self.wf = wf
        src = self.plants.source.values
        pot = np.zeros((H, len(self.plants)))
        for s in ("geothermal", "wind", "solar", "import"):
            m = src == s
            wts = self.cap[m] * self.avail_fc[:, m] + 1e-9
            wts = wts / wts.sum(axis=1, keepdims=True)
            pot[:, m] = self.gen_fc[s].values[:, None] * wts
        m = src == "hydro"
        pot[:, m] = self.cap[m] * self.avail_fc[:, m] * wf
        m = src == "thermal"
        pot[:, m] = self.cap[m] * self.avail_fc[:, m]
        self.plant_pot0 = pot
        self.bus_idx = {b: i for i, b in enumerate(self.net.buses)}
        self.plant_bus = np.array([self.bus_idx[b] for b in self.plants.bus_id])
        self.cell_bus = np.array([self.bus_idx[b] for b in self.cells.bus_id])
        self.plant_cell = {pid: cid for pid, cid in zip(self.plants.plant_id, self.plants.cell_id)}

    # ------------------------------------------------------------------ dispatch / scenario engine
    def run_dispatch(self, plant_pot: np.ndarray | None = None, cell_dem: np.ndarray | None = None,
                     line_derate: dict[str, float] | None = None, battery: dict | None = None,
                     hours: int | None = None) -> dict[str, Any]:
        H = hours or HORIZON_MAX
        pot = (self.plant_pot0 if plant_pot is None else plant_pot)[:H].copy()
        dem_c = (self.cell_dem if cell_dem is None else cell_dem)[:H]
        src = self.plants.source.values
        derate = np.ones(len(self.net.lines))
        for lid, d in (line_derate or {}).items():
            derate[self.net.lines.index[self.net.lines.line_id == lid][0]] = max(0.0, 1 - d)
        # transmission corridor limit for Lake Turkana wind (reduced if the corridor line is derated)
        i_ltwp = int(np.where(self.plants.plant_id.values == "LTWP")[0][0])
        corr_idx = int(self.net.lines.index[self.net.lines.line_id == "LTWP_HUB-SUSWA"][0])
        corridor = LTWP_CORRIDOR_LIMIT_MW * derate[corr_idx]
        corridor_curt = np.clip(pot[:, i_ltwp] - corridor, 0, None)
        pot[:, i_ltwp] -= corridor_curt

        def agg(mask):
            return pot[:, mask].sum(axis=1)
        demand = dem_c.sum(axis=1)
        required = demand * (1 + TLOSS)
        geo_, imp_ = agg(src == "geothermal"), agg(src == "import")
        wind, solar = agg(src == "wind"), agg(src == "solar")
        hyd_av, th_cap = agg(src == "hydro"), agg(src == "thermal")

        def solve(req):
            vre = wind + solar
            imp_u = np.minimum(imp_, np.clip(req - geo_, 0, None))
            geo_u = np.minimum(geo_, req)
            nonflex = geo_u + imp_u
            hmin = np.minimum(0.30 * hyd_av, np.clip(req - nonflex, 0, None))
            excess = np.clip(nonflex + hmin + vre - req, 0, None)
            curt = np.minimum(excess, vre)
            scale = np.where(vre > 0, 1 - curt / np.maximum(vre, 1e-9), 1.0)
            hydro = np.clip(req - nonflex - vre * scale, hmin, hyd_av)
            thermal = np.clip(req - nonflex - vre * scale - hydro, 0, th_cap)
            served = nonflex + vre * scale + hydro + thermal
            self._gf = (np.where(geo_ > 0, geo_u / np.maximum(geo_, 1e-9), 1.0), np.where(imp_ > 0, imp_u / np.maximum(imp_, 1e-9), 1.0))
            return curt, scale, hydro, thermal, np.clip(req - served, 0, None), nonflex + vre * scale

        curt, scale, hydro, thermal, unserved, must = solve(required)
        batt_res = None
        if battery:
            spec = BatterySpec(power_mw=battery["power_mw"], energy_mwh=battery["energy_mwh"],
                               round_trip_eff=battery.get("round_trip_eff", 0.88),
                               reserve_frac=battery.get("reserve_frac", 0.15))
            net_load = required - must
            price = 40 + 120 * (thermal / max(th_cap.max(), 1.0)) + 400 * (unserved > 0)
            batt_res = optimize_battery(net_load, spec, price_per_mwh=price, free_charge_mw=curt)
            if batt_res.get("feasible"):
                adj = np.array(batt_res["charge_mw"]) - np.array(batt_res["discharge_mw"])
                curt, scale, hydro, thermal, unserved, must = solve(required + adj)
        fac = np.ones_like(pot)
        gf, imf = self._gf
        fac[:, src == "geothermal"] = gf[:, None]
        fac[:, src == "import"] = imf[:, None]
        for s, f in (("wind", scale), ("solar", scale)):
            fac[:, src == s] = f[:, None]
        fac[:, src == "hydro"] = (hydro / np.maximum(hyd_av, 1e-9))[:, None]
        fac[:, src == "thermal"] = (thermal / np.maximum(th_cap, 1e-9))[:, None]
        disp = pot * fac
        inj = np.zeros((H, len(self.net.buses)))
        for p in range(len(self.plants)):
            inj[:, self.plant_bus[p]] += disp[:, p]
        for c in range(dem_c.shape[1]):
            inj[:, self.cell_bus[c]] -= dem_c[:, c] * (1 + TLOSS)
        if batt_res and batt_res.get("feasible"):
            inj[:, self.bus_idx.get(battery.get("bus_id", "EMBAKASI"), 0)] += np.array(batt_res["discharge_mw"]) - np.array(batt_res["charge_mw"])
        flows, loading = self.net.flows(inj, derate)
        reserve = (geo_ + imp_ + wind * scale + solar * scale + hyd_av + th_cap - required) / np.maximum(required, 1) * 100
        return {"demand": demand, "required": required, "geo": geo_, "imp": imp_, "wind": wind * scale,
                "solar": solar * scale, "hydro": hydro, "thermal": thermal, "curtailed": curt + corridor_curt,
                "corridor_curtailed": corridor_curt, "unserved": unserved, "reserve_margin_pct": reserve,
                "plant_disp": disp, "flows": flows, "loading": loading, "derate": derate, "battery": batt_res,
                "cell_demand": dem_c, "hydro_avail": hyd_av, "thermal_cap": th_cap}

    def _summarise(self, r: dict) -> dict[str, Any]:
        L = r["loading"]
        lines = self.net.lines
        over = {lines.line_id[i]: int((L[:, i] > 1.0).sum()) for i in range(L.shape[1]) if (L[:, i] > 1.0).any()}
        return {
            "unserved_energy_mwh": round(float(r["unserved"].sum()), 3), "peak_unserved_mw": round(float(r["unserved"].max()), 3),
            "hours_with_unserved": int((r["unserved"] > 1).sum()), "min_reserve_margin_pct": float(r["reserve_margin_pct"].min()),
            "peak_demand_mw": float(r["demand"].max()), "thermal_energy_mwh": float(r["thermal"].sum()),
            "hydro_energy_mwh": float(r["hydro"].sum()), "curtailed_energy_mwh": float(r["curtailed"].sum()),
            "overloaded_line_hours": over, "max_line_loading_pct": {lines.line_id[i]: float(L[:, i].max() * 100) for i in np.argsort(-L.max(axis=0))[:5]},
        }

    def scenario(self, req: dict) -> dict[str, Any]:
        H = int(req.get("horizon_hours", 24))
        pot = self.plant_pot0.copy()
        dem = self.cell_dem.copy()
        src = self.plants.source.values
        applied = []
        for gr in req.get("generation_reductions", []):
            m = src == gr["source"]
            pot[:, m] *= 1 - gr["pct"] / 100.0
            applied.append(f"{gr['source']} output -{gr['pct']:g}%")
        pid_idx = {p: i for i, p in enumerate(self.plants.plant_id)}
        for o in req.get("plant_outages", []):
            if o["plant_id"] not in pid_idx:
                raise ValueError(f"unknown plant_id {o['plant_id']}")
            a, b = o.get("start_hour", 0), o.get("start_hour", 0) + o.get("duration_h", H)
            pot[a:b, pid_idx[o["plant_id"]]] *= 1 - o.get("fraction", 1.0)
            applied.append(f"outage {o['plant_id']} h{a}-{b} ({o.get('fraction', 1.0):.0%})")
        di = req.get("demand_increase_pct", 0.0)
        if di:
            reg = req.get("demand_region")
            m = (self.cells.region.values == reg) if reg else np.ones(dem.shape[1], bool)
            dem[:, m] *= 1 + di / 100.0
            applied.append(f"demand +{di:g}% ({reg or 'national'})")
        derate = {}
        for ld in req.get("line_derates", []):
            if ld["line_id"] not in set(self.net.lines.line_id):
                raise ValueError(f"unknown line_id {ld['line_id']}")
            derate[ld["line_id"]] = ld["derate_pct"] / 100.0
            applied.append(f"line {ld['line_id']} derated {ld['derate_pct']}%")
        batt = req.get("battery")
        if req.get("curtail_renewables_pct"):
            f = 1 - req["curtail_renewables_pct"] / 100.0
            pot[:, np.isin(src, ["wind", "solar"])] *= f
            applied.append(f"manual renewable curtailment {req['curtail_renewables_pct']}%")
        if batt:
            applied.append(f"battery {batt['power_mw']} MW / {batt['energy_mwh']} MWh at {batt.get('bus_id', 'EMBAKASI')}")
        res = self.run_dispatch(pot, dem, derate, batt, hours=H)
        base = self.run_dispatch(None, None, None, None, hours=H)
        sb, ss = self._summarise(base), self._summarise(res)
        delta = {k: round(ss[k] - sb[k], 3) for k in ("unserved_energy_mwh", "peak_unserved_mw", "thermal_energy_mwh",
                                            "hydro_energy_mwh", "curtailed_energy_mwh", "min_reserve_margin_pct")}
        warns = []
        if ss["unserved_energy_mwh"] > sb["unserved_energy_mwh"] + 1:
            warns.append(f"Scenario adds {delta['unserved_energy_mwh']:.0f} MWh of unserved energy versus baseline.")
        if ss["min_reserve_margin_pct"] < 5:
            warns.append("Minimum reserve margin falls below 5% in this scenario.")
        for lid, hrs in ss["overloaded_line_hours"].items():
            warns.append(f"Line {lid} exceeds assumed capacity for {hrs} h (DC approximation).")
        times = [_iso(t) for t in self.times_fc[:H]]
        return {
            "scenario_name": req.get("name", "custom"), "applied_changes": applied, "horizon_hours": H,
            "label": SCENARIO_LABEL, "data_origin": "forecast+synthetic",
            "baseline": sb, "scenario": ss, "delta_vs_baseline": delta, "warnings": warns,
            "series": {"time": times, "demand_mw": np.round(res["demand"], 2).tolist(), "unserved_mw": np.round(res["unserved"], 2).tolist(),
                       "baseline_unserved_mw": np.round(base["unserved"], 2).tolist(), "thermal_mw": np.round(res["thermal"], 2).tolist(),
                       "hydro_mw": np.round(res["hydro"], 2).tolist(), "reserve_margin_pct": np.round(res["reserve_margin_pct"], 2).tolist(),
                       "curtailed_mw": np.round(res["curtailed"], 2).tolist()},
            "line_loading_max_pct": {self.net.lines.line_id[i]: float(res["loading"][:, i].max() * 100) for i in range(len(self.net.lines))},
            "battery": res["battery"], "model_version": self.fb.version, "generated_at": self._now(),
            "approval_required": True,
            "note": "Recommendations from this scenario are advisory; any operational action requires authorised human approval.",
        }

    # ------------------------------------------------------------------ views
    def _now(self) -> str:
        return dt.datetime.now(dt.timezone(dt.timedelta(hours=3))).isoformat(timespec="seconds")

    def provenance(self) -> dict:
        return {**self.world.provenance, "last_observed_hour": _iso(self.world.times[self.world.n_hist - 1]),
                "labels": {"history": "synthetic", "forecast": "model forecast over synthetic history", "live": "none connected"}}

    def overview(self) -> dict:
        w, n = self.world, self.world.n_hist
        last = w.system.iloc[n - 1]
        b = self.baseline
        t24 = slice(0, 24)
        pk = int(np.argmax(b["demand"][t24]))
        mix = {k: float(last[f"{k}_mw"]) for k in ("geothermal", "hydro", "wind", "solar", "thermal", "import")}
        rm = ml.reg_metrics
        return {
            "as_of": _iso(w.times[n - 1]), "data_origin": "synthetic", "provenance": self.provenance(),
            "current": {"demand_mw": float(last.total_demand_mw), "generation_mix_mw": mix,
                        "renewable_share_pct": float((mix["geothermal"] + mix["hydro"] + mix["wind"] + mix["solar"]) / max(sum(mix.values()), 1) * 100),
                        "unserved_mw": float(last.unserved_mw), "water_factor": float(last.water_factor)},
            "next_24h": {"peak_demand_mw": float(b["demand"][t24][pk]), "peak_time": _iso(self.times_fc[pk]),
                         "demand_interval_80": self._demand_interval(pk),
                         "min_reserve_margin_pct": float(b["reserve_margin_pct"][t24].min()),
                         "hours_with_unserved": int((b["unserved"][t24] > 1).sum()),
                         "curtailed_mwh": float(b["curtailed"][t24].sum())},
            "assets": {"units_flagged_high_risk": int(self.maint_risk.flagged.sum()),
                       "telemetry_anomalies_72h": int(len(self.anomalies)),
                       "feeders_flagged_losses": int(self.lb.latest.flagged.sum()),
                       "lines_over_80pct_next_24h": int((b["loading"][t24].max(axis=0) > 0.8).sum())},
            "model_versions": self.model_versions,
            "demand_forecast_holdout_mape_24h_pct": self.fb.metrics["demand"]["h24"]["national"]["model"]["mape_pct"],
        }

    def _demand_interval(self, pk: int) -> list[float]:
        h = ml.lead_to_horizon(pk + 1)
        q10, q90 = self.fb.demand_rel_err_q[h]
        v = float(self.baseline["demand"][pk])
        return [v / (1 + q90), v / (1 + q10)]

    def hist_series(self, hours: int = 168) -> dict:
        w, n = self.world, self.world.n_hist
        s = w.system.iloc[n - hours:n]
        return {"time": [_iso(t) for t in s.index], "demand_mw": s.total_demand_mw.round(1).tolist(),
                **{k: s[f"{k}_mw"].round(1).tolist() for k in ("geothermal", "hydro", "wind", "solar", "thermal", "import")},
                "unserved_mw": s.unserved_mw.round(1).tolist(), "data_origin": "synthetic"}

    def demand_view(self) -> dict:
        w = self.world
        tot = self.dem_fc.sum(axis=1)
        lo, hi = [], []
        for i, v in enumerate(tot.values):
            q10, q90 = self.fb.demand_rel_err_q[ml.lead_to_horizon(i + 1)]
            lo.append(float(v / (1 + q90)))
            hi.append(float(v / (1 + q10)))
        return {"data_origin": "forecast", "model_version": self.fb.version, "time": [_iso(t) for t in tot.index],
                "national_mw": tot.round(1).tolist(), "lo80_mw": np.round(lo, 1).tolist(), "hi80_mw": np.round(hi, 1).tolist(),
                "regional_mw": {r: self.dem_fc[r].round(1).tolist() for r in geo.REGION_IDS},
                "history": self.hist_series(168), "metrics": self.fb.metrics["demand"],
                "limitations": ["Regional demand is modelled directly; cell-level demand uses static within-region shares (illustrative).",
                                "Weather inputs are synthetic forecasts with horizon-dependent noise, not NWP output."]}

    def generation_view(self) -> dict:
        idx = [_iso(t) for t in self.times_fc]
        return {"data_origin": "forecast", "model_version": self.fb.version, "time": idx,
                "mean_mw": {k: v.round(1).tolist() for k, v in self.gen_fc.items()},
                "lo80_mw": {k: v.round(1).tolist() for k, v in self.gen_lo.items()},
                "hi80_mw": {k: v.round(1).tolist() for k, v in self.gen_hi.items()},
                "history": self.hist_series(168), "metrics": self.fb.metrics["generation"],
                "plants": self.plants[["plant_id", "name", "source", "capacity_mw", "region", "lon", "lat"]].to_dict("records"),
                "installed_capacity_mw": self.plants.groupby("source").capacity_mw.sum().to_dict(),
                "dispatch_next_72h": {k: np.round(self.baseline[k], 1).tolist() for k in ("geo", "hydro", "wind", "solar", "thermal", "imp", "unserved", "reserve_margin_pct")},
                "note": "Hydro and thermal are re-dispatched by a merit-order engine from forecast demand and variable supply; ML forecasts for hydro/thermal are reported in metrics."}

    def maintenance_view(self) -> dict:
        mr = self.maint_risk.copy()
        mr["plant_name"] = mr.plant_id.map(dict(zip(self.plants.plant_id, self.plants.name)))
        return {"data_origin": "synthetic+model", "model_version": self.mb.version, "as_of": _iso(self.world.times[self.world.n_hist - 1]),
                "units": mr.round({"risk_probability": 3, "vib_mean": 2, "temp_c": 1, "pressure_bar": 2, "flow_rate": 1, "op_hours": 0}).to_dict("records"),
                "decision_threshold": self.mb.threshold, "metrics": self.mb.metrics, "feature_importance": self.mb.importances,
                "label_definition": "fault within next 14 days", "limitations": ["Trained on synthetic latent-health faults; real-world performance unknown."]}

    def anomalies_view(self) -> dict:
        a = self.anomalies.copy()
        a["time"] = a.time.map(_iso)
        return {"data_origin": "synthetic+model", "model_version": self.ab.version, "window_hours": 72,
                "events": a.round({"score": 3}).to_dict("records"), "metrics": self.ab.metrics,
                "note": "is_anomaly is the injected ground-truth label for validation (unavailable in live data)."}

    def losses_view(self) -> dict:
        lt = self.lb.latest
        by_cell = lt.groupby("cell_id").agg(input_mwh=("input_mwh", "sum"), metered_mwh=("metered_mwh", "sum"),
                                            ntl_mwh=("ntl_mwh_est", "sum"), tech_mwh=("tech_mwh_est", "sum"), flagged=("flagged", "sum"))
        by_reg = lt.groupby("region").agg(input_mwh=("input_mwh", "sum"), metered_mwh=("metered_mwh", "sum"),
                                          ntl_mwh=("ntl_mwh_est", "sum"), tech_mwh=("tech_mwh_est", "sum"), flagged=("flagged", "sum"))
        by_reg["loss_pct"] = (1 - by_reg.metered_mwh / by_reg.input_mwh) * 100
        by_reg["ntl_pct"] = by_reg.ntl_mwh / by_reg.input_mwh * 100
        by_reg["tech_pct"] = by_reg.tech_mwh / by_reg.input_mwh * 100
        top = lt[lt.flagged == 1].sort_values("ntl_mwh_est", ascending=False).head(25)
        return {"data_origin": "synthetic+model", "model_version": self.lb.version, "month_index": int(lt.month.max()),
                "regions": by_reg.round(2).reset_index().to_dict("records"),
                "flagged_feeders": top[["feeder_id", "region", "cell_id", "ntl_est_pct", "ntl_mwh_est", "anomaly_score", "peer_z"]].round(3).to_dict("records"),
                "totals": {"input_mwh": float(lt.input_mwh.sum()), "metered_mwh": float(lt.metered_mwh.sum()),
                           "loss_pct": float((1 - lt.metered_mwh.sum() / lt.input_mwh.sum()) * 100),
                           "ntl_mwh": float(lt.ntl_mwh_est.sum()), "tech_mwh": float(lt.tech_mwh_est.sum())},
                "metrics": self.lb.metrics,
                "caveat": "NTL in this dataset is injected with large effect sizes, so the simple threshold rule separates classes perfectly; expect lower performance on real data."}

    def curtailment_view(self) -> dict:
        ev = self.world.curtailment_events
        ev = ev[ev.time >= self.world.times[self.world.n_hist - 1] - pd.Timedelta(days=30)]
        by_plant = ev.groupby(["plant_id", "reason"]).curtailed_mw.agg(["sum", "count", "max"]).reset_index()
        hour = ev.groupby(ev.time.dt.hour).curtailed_mw.sum()
        names = dict(zip(self.plants.plant_id, self.plants.name))
        return {"data_origin": "synthetic", "window_days": 30, "total_curtailed_mwh": float(ev.curtailed_mw.sum()),
                "by_plant": [{"plant_id": r.plant_id, "plant": names[r.plant_id], "reason": r.reason, "mwh": float(r["sum"]),
                              "hours": int(r["count"]), "max_mw": float(r["max"])} for _, r in by_plant.iterrows()],
                "by_hour_of_day_mwh": {int(h): float(v) for h, v in hour.items()},
                "forecast_next_24h_mwh": float(self.baseline["curtailed"][:24].sum()),
                "opportunity": "Curtailed energy is storable: see /storage/optimize with free-charge from curtailment."}

    def grid_view(self, hour: int = 0) -> dict:
        b = self.baseline
        lines = self.net.lines
        L = b["loading"]
        return {"data_origin": "forecast+synthetic", "label": SCENARIO_LABEL, "hour": hour,
                "buses": self.world.substations[["bus_id", "name", "lon", "lat", "kv"]].to_dict("records"),
                "lines": [{"line_id": lines.line_id[i], "from_bus": lines.from_bus[i], "to_bus": lines.to_bus[i],
                           "capacity_mw": float(lines.capacity_mw[i]), "flow_mw": float(b["flows"][hour, i]),
                           "loading_pct": float(L[hour, i] * 100), "max_loading_pct_72h": float(L[:, i].max() * 100),
                           "hours_over_80pct": int((L[:, i] > 0.8).sum()), "hours_over_100pct": int((L[:, i] > 1.0).sum())}
                          for i in range(len(lines))],
                "bottlenecks": [{"line_id": lines.line_id[i], "max_loading_pct": float(L[:, i].max() * 100)}
                                for i in np.argsort(-L.max(axis=0))[:5] if L[:, i].max() > 0.7],
                "incidents_recent": self.world.grid_incidents.tail(8).assign(time=lambda d: d.time.map(_iso)).to_dict("records")}

    def n1_screen(self, hour: int | None = None) -> dict:
        """N-1 contingency screening on the DC model at the peak-demand hour (or a given hour)."""
        b = self.baseline
        h = int(np.argmax(b["demand"])) if hour is None else int(hour)
        inj = np.zeros(len(self.net.buses))
        for p in range(len(self.plants)):
            inj[self.plant_bus[p]] += b["plant_disp"][h, p]
        for c in range(b["cell_demand"].shape[1]):
            inj[self.cell_bus[c]] -= b["cell_demand"][h, c] * (1 + TLOSS)
        lines = self.net.lines
        results = []
        for k in range(len(lines)):
            keep = lines.drop(index=k).reset_index(drop=True)
            try:
                net = DCNetwork(self.world.substations, keep)
            except np.linalg.LinAlgError:
                results.append({"outage": lines.line_id[k], "islanding": True, "overloads": []})
                continue
            f, load = net.flows(inj[None, :])
            ov = [{"line_id": keep.line_id[i], "loading_pct": float(load[0, i] * 100)} for i in range(len(keep)) if load[0, i] > 1.0]
            results.append({"outage": lines.line_id[k], "islanding": False,
                            "overloads": sorted(ov, key=lambda d: -d["loading_pct"])[:4]})
        crit = [r for r in results if r["islanding"] or r["overloads"]]
        return {"hour": h, "time": _iso(self.times_fc[h]), "label": SCENARIO_LABEL, "n_contingencies": len(results),
                "contingencies_with_violations": len(crit), "violations": crit, "data_origin": "forecast+synthetic"}

    def climate_view(self) -> dict:
        w = self.world
        n = w.n_hist
        wt = w.weather
        last = wt[wt.time >= w.times[n - 24]]
        cur = last[last.time < w.times[n]].groupby("region")[["temp_c", "ghi_wm2", "wind_ms", "cloud", "rain_mm"]].mean().round(2)
        fut = wt[(wt.time >= w.times[n]) & (wt.time < w.times[n + 24])].groupby("region")[["temp_c", "ghi_wm2", "wind_ms", "cloud", "rain_mm"]].mean().round(2)
        ser = wt[wt.region == "NAIROBI"].set_index("time").iloc[n - 168:n]
        return {"data_origin": "synthetic", "last_24h_mean": cur.reset_index().to_dict("records"),
                "next_24h_mean_synthetic_forecast": fut.reset_index().to_dict("records"),
                "water_factor_history": [round(float(x), 3) for x in w.system.water_factor.iloc[n - 24 * 60:n:24]],
                "water_factor_dates": [_iso(t)[:10] for t in w.times[n - 24 * 60:n:24]],
                "note": "Weather is synthetic. Open-Meteo / NASA POWER connectors are not enabled in this build."}

    # ------------------------------------------------------------------ usage analysis
    def usage_view(self) -> dict:
        w, n = self.world, self.world.n_hist
        d = w.demand.iloc[:n]
        last30, prev30 = d.iloc[-720:], d.iloc[-1440:-720]
        out = []
        lr = self.lb.latest.groupby("region").agg(i=("input_mwh", "sum"), m=("metered_mwh", "sum"))
        reg_cells = self.cells.groupby("region").cell_id.apply(list)
        local_gen = {}
        for r in geo.REGION_IDS:
            cids = set(reg_cells.get(r, []))
            m = np.array([self.plant_cell[p] in cids for p in self.plants.plant_id])
            local_gen[r] = float(self.baseline["plant_disp"][:24][:, m].sum(axis=1).mean()) if m.any() else 0.0
        for r in geo.REGION_IDS:
            s = last30[r]
            demand24 = float(self.dem_fc[r].iloc[:24].mean())
            out.append({"region": r, "name": next(x.name for x in geo.REGIONS if x.id == r),
                        "energy_gwh_30d": float(s.sum() / 1000), "avg_mw": float(s.mean()), "peak_mw": float(s.max()),
                        "peak_hour": int(s.groupby(s.index.hour).mean().idxmax()), "load_factor_pct": float(s.mean() / s.max() * 100),
                        "growth_vs_prev_30d_pct": float((s.sum() / prev30[r].sum() - 1) * 100),
                        "share_of_national_pct": float(s.sum() / last30.sum().sum() * 100),
                        "loss_pct": float((1 - lr.loc[r, "m"] / lr.loc[r, "i"]) * 100),
                        "forecast_avg_mw_24h": demand24, "local_generation_mw_24h_avg": local_gen[r],
                        "import_dependence_pct": float(max(0.0, demand24 - local_gen[r]) / max(demand24, 1) * 100)})
        prof = {r: d[r].iloc[-720:].groupby(d.index[-720:].hour).mean().round(1).tolist() for r in geo.REGION_IDS}
        return {"data_origin": "synthetic", "regions": out, "hourly_profile_mw": prof, "window": "last 30 days"}

    # ------------------------------------------------------------------ map layer
    def _cell_static(self) -> pd.DataFrame:
        lt = self.lb.latest.groupby("cell_id").agg(i=("input_mwh", "sum"), m=("metered_mwh", "sum"), ntl=("ntl_mwh_est", "sum"),
                                                   tech=("tech_mwh_est", "sum"), fl=("flagged", "sum"))
        df = pd.DataFrame(index=self.cells.cell_id)
        df["loss_pct"] = (1 - lt.m / lt.i) * 100
        df["ntl_pct"] = lt.ntl / lt.i * 100
        df["tech_pct"] = lt.tech / lt.i * 100
        df["flagged_feeders"] = lt.fl
        risk = self.maint_risk.copy()
        risk["cell_id"] = risk.plant_id.map(self.plant_cell)
        df["maintenance_risk"] = risk.groupby("cell_id").risk_probability.max() * 100
        ev = self.world.curtailment_events
        ev = ev[ev.time >= self.world.times[self.world.n_hist - 1] - pd.Timedelta(days=30)].copy()
        ev["cell_id"] = ev.plant_id.map(self.plant_cell)
        df["curtailed_mwh_30d"] = ev.groupby("cell_id").curtailed_mw.sum()
        return df.fillna(0.0)

    METRICS = {
        "demand_mw": "Demand (MW)", "generation_mw": "Local generation (MW)", "net_flow_mw": "Net export (+) / import (-) (MW)",
        "import_dependence_pct": "Import dependence (%)", "loss_pct": "Total distribution loss (%)", "ntl_pct": "Non-technical loss (%)",
        "tech_pct": "Technical loss (%)", "maintenance_risk": "Max unit failure risk (%)", "renewable_mw": "Local renewable generation (MW)",
        "curtailed_mwh_30d": "Curtailed energy, 30 d (MWh)", "line_loading_pct": "Adjacent line loading (%)",
        "composite_risk": "Composite grid-stress index (0-100)",
    }

    def map_cells(self, hour: int = 0) -> dict:
        hour = int(np.clip(hour, 0, HORIZON_MAX - 1))
        b = self.baseline
        st = self._cell_static()
        dem = b["cell_demand"][hour]
        disp = b["plant_disp"][hour]
        src = self.plants.source.values
        gen = np.zeros(len(self.cells))
        ren = np.zeros(len(self.cells))
        cidx = {c: i for i, c in enumerate(self.cells.cell_id)}
        for p in range(len(self.plants)):
            ci = cidx[self.plants.cell_id.iloc[p]]
            gen[ci] += disp[p]
            if src[p] in ("geothermal", "hydro", "wind", "solar"):
                ren[ci] += disp[p]
        bus_load = np.zeros(len(self.net.buses))
        for i, bus in enumerate(self.net.buses):
            inc = [k for k in range(len(self.net.lines)) if self.net.lines.from_bus[k] == bus or self.net.lines.to_bus[k] == bus]
            bus_load[i] = b["loading"][hour, inc].max() * 100 if inc else 0.0
        line_load = bus_load[self.cell_bus]
        demand_with_loss = dem * (1 + TLOSS)
        net = gen - demand_with_loss
        imp_dep = np.clip(demand_with_loss - gen, 0, None) / np.maximum(demand_with_loss, 1e-6) * 100
        s = st.reset_index(drop=True)
        def norm(a, lo, hi):
            return np.clip((np.asarray(a, float) - lo) / (hi - lo), 0, 1)
        comp = 100 * (0.30 * norm(s.loss_pct, 8, 30) + 0.25 * norm(s.maintenance_risk, 0, 80) + 0.20 * norm(imp_dep, 0, 100)
                      + 0.15 * norm(line_load, 30, 100) + 0.10 * norm(s.flagged_feeders, 0, 3))
        feats = []
        for i, c in enumerate(self.cells.itertuples()):
            props = {"cell_id": c.cell_id, "region": c.region, "bus_id": c.bus_id, "demand_mw": round(float(dem[i]), 3),
                     "generation_mw": round(float(gen[i]), 2), "net_flow_mw": round(float(net[i]), 2),
                     "import_dependence_pct": round(float(imp_dep[i]), 1), "loss_pct": round(float(s.loss_pct[i]), 2),
                     "ntl_pct": round(float(s.ntl_pct[i]), 2), "tech_pct": round(float(s.tech_pct[i]), 2),
                     "maintenance_risk": round(float(s.maintenance_risk[i]), 1), "renewable_mw": round(float(ren[i]), 2),
                     "curtailed_mwh_30d": round(float(s.curtailed_mwh_30d[i]), 1), "line_loading_pct": round(float(line_load[i]), 1),
                     "composite_risk": round(float(comp[i]), 1), "flagged_feeders": int(s.flagged_feeders[i])}
            feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [c.polygon]}})
        maxes = {m: float(max(abs(f["properties"][m]) for f in feats)) for m in self.METRICS}
        return {"type": "FeatureCollection", "features": feats,
                "meta": {"hour": hour, "time": _iso(self.times_fc[hour]), "metrics": self.METRICS, "max": maxes,
                         "data_origin": "forecast+synthetic",
                         "geometry_note": "Hex cells over an approximate Kenya outline; illustrative, not administrative boundaries.",
                         "composite_formula": "100*(0.30*loss + 0.25*maint_risk + 0.20*import_dependence + 0.15*line_loading + 0.10*flagged_feeders), each min-max normalised",
                         "model_version": self.fb.version}}

    def map_assets(self, hour: int = 0) -> dict:
        b = self.baseline
        plants = []
        risk = self.maint_risk.groupby("plant_id").risk_probability.max()
        for i, p in enumerate(self.plants.itertuples()):
            plants.append({"plant_id": p.plant_id, "name": p.name, "source": p.source, "capacity_mw": float(p.capacity_mw),
                           "dispatch_mw": round(float(b["plant_disp"][hour, i]), 1), "lon": p.lon, "lat": p.lat,
                           "max_unit_risk": round(float(risk.get(p.plant_id, 0.0)), 3)})
        L = b["loading"][hour]
        lines = self.net.lines
        bx = {r.bus_id: (r.lon, r.lat) for r in self.world.substations.itertuples()}
        lf = [{"type": "Feature", "properties": {"line_id": lines.line_id[i], "loading_pct": round(float(L[i] * 100), 1),
                                                  "flow_mw": round(float(b["flows"][hour, i]), 1), "capacity_mw": float(lines.capacity_mw[i])},
               "geometry": {"type": "LineString", "coordinates": [list(bx[lines.from_bus[i]]), list(bx[lines.to_bus[i]])]}}
              for i in range(len(lines))]
        outline = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in geo.KENYA_OUTLINE]]}}
        return {"plants": plants, "substations": self.world.substations[["bus_id", "name", "lon", "lat", "kv"]].to_dict("records"),
                "lines": {"type": "FeatureCollection", "features": lf}, "outline": outline, "hour": hour}

    def map_timeline(self) -> dict:
        b = self.baseline
        return {"time": [_iso(t) for t in self.times_fc], "demand_mw": np.round(b["demand"], 1).tolist(),
                "generation_mw": np.round(b["required"] - b["unserved"], 1).tolist(), "unserved_mw": np.round(b["unserved"], 1).tolist(),
                "max_line_loading_pct": np.round(b["loading"].max(axis=1) * 100, 1).tolist()}

    # ------------------------------------------------------------------ storage
    def storage_optimize(self, power_mw: float, energy_mwh: float, round_trip_eff: float = 0.88, reserve_frac: float = 0.15,
                         horizon_hours: int = 24, degradation_cost: float = 4.0) -> dict:
        b = self.baseline
        H = horizon_hours
        must = b["geo"][:H] + b["imp"][:H] + b["wind"][:H] + b["solar"][:H]
        net_load = b["required"][:H] - must
        price = 40 + 120 * (b["thermal"][:H] / max(b["thermal_cap"].max(), 1)) + 400 * (b["unserved"][:H] > 0)
        spec = BatterySpec(power_mw=power_mw, energy_mwh=energy_mwh, round_trip_eff=round_trip_eff, reserve_frac=reserve_frac,
                           degradation_cost_per_mwh=degradation_cost)
        r = optimize_battery(net_load, spec, price_per_mwh=price, free_charge_mw=b["curtailed"][:H])
        r.update({"time": [_iso(t) for t in self.times_fc[:H]], "objective": "minimise peak residual load (served by hydro+thermal) plus cost proxy",
                  "load_definition": "required generation minus geothermal, import and (post-curtailment) wind+solar",
                  "curtailed_mw_available": b["curtailed"][:H].tolist(), "data_origin": "forecast+synthetic", "model_version": self.fb.version})
        return r

    def storage_compare(self, options: list[dict]) -> dict:
        rows = []
        for o in options:
            r = self.storage_optimize(o["power_mw"], o["energy_mwh"])
            rows.append({"power_mw": o["power_mw"], "energy_mwh": o["energy_mwh"], "status": r["status"],
                         "peak_reduction_mw": r.get("peak_reduction_mw"), "peak_after_mw": r.get("peak_after_mw"),
                         "energy_discharged_mwh": r.get("energy_discharged_mwh"), "cycles": r.get("equivalent_full_cycles"),
                         "degradation_cost_usd": r.get("degradation_cost_usd"), "schedule_valid": r.get("schedule_valid")})
        return {"options": rows, "data_origin": "forecast+synthetic"}

    def metrics_summary(self) -> dict:
        return {"demand": self.fb.metrics["demand"], "generation": self.fb.metrics["generation"], "maintenance": self.mb.metrics,
                "anomaly": self.ab.metrics, "loss": self.lb.metrics, "model_versions": self.model_versions, "trained_at": self.trained_at,
                "evaluation_note": "All metrics are measured on held-out windows of SYNTHETIC data; they do not predict real-world performance."}


_instance: PowerIQ | None = None
_lock = threading.Lock()


def get_service() -> PowerIQ:
    global _instance
    with _lock:
        if _instance is None:
            _instance = PowerIQ()
        return _instance
