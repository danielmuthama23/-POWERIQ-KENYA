"""POWERIQ agents: seven specialists plus a Chief Energy Intelligence Agent.

Every agent has a defined responsibility, typed I/O, an allow-list of read-only tools, a minimum role, error handling,
evidence requirements (each finding is tagged measured / prediction / assumption / recommendation and carries a source),
execution logging, and a confidence value with an explicit definition. No agent has a tool that can act on infrastructure.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field

from . import geo
from .rag import get_kb
from .security import AuditLog, ROLE_LEVEL, get_settings
from .service import PowerIQ, _iso

Kind = Literal["measured", "prediction", "assumption", "recommendation"]


class Finding(BaseModel):
    kind: Kind
    statement: str
    source: str
    value: Any | None = None


class AgentInput(BaseModel):
    question: str | None = Field(default=None, max_length=1000)
    params: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    agent: str
    status: Literal["ok", "error", "insufficient_evidence"]
    summary: str
    findings: list[Finding] = []
    recommendations: list[str] = []
    citations: list[dict] = []
    confidence: float | None = None
    confidence_definition: str = ""
    uncertainty: str = ""
    data_origin: str = "synthetic"
    model_version: str = ""
    limitations: list[str] = []
    tools_used: list[str] = []
    duration_ms: float = 0.0
    timestamp: str = ""
    error: str | None = None


# ------------------------------------------------------------------ MCP-style read-only tool registry
class Tool:
    def __init__(self, name: str, description: str, fn: Callable[..., Any], schema: dict):
        self.name, self.description, self.fn, self.schema = name, description, fn, schema
        self.read_only = True


def build_tools(svc: PowerIQ) -> dict[str, Tool]:
    obj = {"type": "object", "properties": {}, "additionalProperties": False}
    def t(name, desc, fn, schema=None):
        return Tool(name, desc, fn, schema or obj)
    hour_schema = {"type": "object", "properties": {"hour": {"type": "integer", "minimum": 0, "maximum": 71}}}
    tools = [
        t("get_overview", "National snapshot: demand, mix, reserve, asset flags.", svc.overview),
        t("get_demand_forecast", "72 h demand forecast with 80% intervals and holdout metrics.", svc.demand_view),
        t("get_generation_forecast", "72 h generation forecast by source with metrics.", svc.generation_view),
        t("get_maintenance_risk", "Unit failure-risk ranking with drivers.", svc.maintenance_view),
        t("get_anomalies", "Recent telemetry anomalies.", svc.anomalies_view),
        t("get_loss_analysis", "Technical and non-technical loss analysis.", svc.losses_view),
        t("get_curtailment", "Renewable curtailment summary.", svc.curtailment_view),
        t("get_grid_status", "Line loading and bottlenecks (DC model).", svc.grid_view, hour_schema),
        t("run_n1_screen", "N-1 contingency screening on the DC model.", svc.n1_screen, hour_schema),
        t("get_climate", "Weather and water-factor summary.", svc.climate_view),
        t("get_usage", "Regional usage analytics.", svc.usage_view),
        t("run_scenario", "Run an advisory what-if scenario (no actuation).", lambda **kw: svc.scenario(kw),
          {"type": "object", "properties": {"name": {"type": "string"}, "generation_reductions": {"type": "array"}}}),
        t("optimize_storage", "Battery schedule optimisation (advisory).", svc.storage_optimize,
          {"type": "object", "properties": {"power_mw": {"type": "number"}, "energy_mwh": {"type": "number"}}, "required": ["power_mw", "energy_mwh"]}),
        t("search_knowledge", "Retrieve cited engineering knowledge.", lambda question, category=None, k=3: get_kb().query(question, k=k, category=category),
          {"type": "object", "properties": {"question": {"type": "string"}, "category": {"type": "string"}}, "required": ["question"]}),
    ]
    return {x.name: x for x in tools}


class Agent:
    name = "agent"
    responsibility = ""
    tools: list[str] = []
    min_role = "viewer"
    kb_category: str | None = None

    def __init__(self, svc: PowerIQ, tools: dict[str, Tool]):
        self.svc, self._tools = svc, tools

    def call(self, tool: str, used: list[str], **kw):
        if tool not in self.tools:
            raise PermissionError(f"{self.name} is not permitted to call tool '{tool}'")
        used.append(tool)
        return self._tools[tool].fn(**kw)

    def info(self) -> dict:
        return {"name": self.name, "responsibility": self.responsibility, "tools": self.tools, "min_role": self.min_role,
                "input_schema": AgentInput.model_json_schema(), "output_schema": AgentOutput.model_json_schema()}

    def knowledge(self, used: list[str], query: str) -> list[dict]:
        r = self.call("search_knowledge", used, question=query, category=self.kb_category)
        return [{k: x[k] for k in ("citation", "title", "source", "score")} | {"excerpt": x["text"][:220]} for x in r["results"]]

    def run(self, inp: AgentInput) -> AgentOutput:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


def _conf_from_mape(mape: float) -> float:
    return float(np.clip(1 - mape / 10.0, 0.3, 0.95))


class DemandAgent(Agent):
    name = "demand_forecasting"
    responsibility = "Forecast national and regional electricity demand and peak consumption with uncertainty."
    tools = ["get_demand_forecast", "get_overview", "search_knowledge"]

    def run(self, inp):
        u: list[str] = []
        d = self.call("get_demand_forecast", u)
        o = self.call("get_overview", u)
        n24 = d["national_mw"][:24]
        pk = int(np.argmax(n24))
        m = d["metrics"]["h24"]["national"]
        f = [Finding(kind="measured", statement=f"Latest observed national demand was {o['current']['demand_mw']:.0f} MW at {o['as_of']}.", source="synthetic history"),
             Finding(kind="prediction", statement=f"Predicted 24 h peak is {n24[pk]:.0f} MW at {d['time'][pk]} (80% interval {d['lo80_mw'][pk]:.0f}-{d['hi80_mw'][pk]:.0f} MW).",
                     source=d["model_version"], value=n24[pk]),
             Finding(kind="prediction", statement=f"Predicted mean demand over the next 24 h is {np.mean(n24):.0f} MW; over 72 h {np.mean(d['national_mw']):.0f} MW.", source=d["model_version"])]
        reg_peak = sorted(((r, max(v[:24])) for r, v in d["regional_mw"].items()), key=lambda x: -x[1])[:3]
        f.append(Finding(kind="prediction", statement="Largest regional 24 h peaks: " + ", ".join(f"{r} {p:.0f} MW" for r, p in reg_peak) + ".", source=d["model_version"]))
        f.append(Finding(kind="measured", statement=f"Holdout 24 h-ahead national MAPE is {m['model']['mape_pct']:.2f}% (same-hour-last-week baseline {m['baseline_same_hour_last_week']['mape_pct']:.2f}%).", source="holdout evaluation (last 28 days, synthetic)"))
        return AgentOutput(agent=self.name, status="ok", summary=f"Peak {n24[pk]:.0f} MW expected at {d['time'][pk][11:16]}.", findings=f,
                           citations=self.knowledge(u, "demand forecasting peak consumption"), confidence=_conf_from_mape(m["model"]["mape_pct"]),
                           confidence_definition="1 - holdout 24h MAPE/10, clipped to [0.3, 0.95]",
                           uncertainty=f"80% interval at peak: {d['lo80_mw'][pk]:.0f}-{d['hi80_mw'][pk]:.0f} MW",
                           data_origin="forecast", model_version=d["model_version"], limitations=d["limitations"], tools_used=u)


class GenerationAgent(Agent):
    name = "generation_intelligence"
    responsibility = "Forecast generation by source, assess reserve margin and explain generation-demand gaps."
    tools = ["get_generation_forecast", "get_overview", "get_curtailment", "search_knowledge"]
    kb_category = "renewables"

    def run(self, inp):
        u: list[str] = []
        g = self.call("get_generation_forecast", u)
        o = self.call("get_overview", u)
        c = self.call("get_curtailment", u)
        disp = g["dispatch_next_72h"]
        res = np.array(disp["reserve_margin_pct"])
        un = np.array(disp["unserved"])
        worst = int(np.argmin(res[:24]))
        f = [Finding(kind="measured", statement="Current mix (MW): " + ", ".join(f"{k} {v:.0f}" for k, v in o["current"]["generation_mix_mw"].items()) + f"; renewables {o['current']['renewable_share_pct']:.0f}%.", source="synthetic history"),
             Finding(kind="prediction", statement=f"Lowest 24 h reserve margin is {res[:24].min():.1f}% at {g['time'][worst]}.", source=g["model_version"], value=float(res[:24].min())),
             Finding(kind="prediction", statement=f"Hours with unserved energy in next 72 h: {int((un > 1).sum())}.", source="merit-order dispatch over ML forecasts")]
        if (un[:24] > 1).any() or res[:24].min() < 10:
            sys = self.svc.baseline
            f.append(Finding(kind="prediction", statement=(
                f"Gap driver at the tightest hour: demand {sys['demand'][worst]:.0f} MW, hydro available {sys['hydro_avail'][worst]:.0f} MW (water factor {self.svc.wf:.2f}), "
                f"thermal available {sys['thermal_cap'][worst]:.0f} MW, wind+solar {sys['wind'][worst] + sys['solar'][worst]:.0f} MW."), source="dispatch engine"))
        f.append(Finding(kind="measured", statement=f"Curtailed energy in the last 30 days: {c['total_curtailed_mwh']:.0f} MWh; forecast next 24 h: {c['forecast_next_24h_mwh']:.0f} MWh.", source="synthetic curtailment events"))
        top = max(c["by_plant"], key=lambda r: r["mwh"], default=None)
        if top:
            f.append(Finding(kind="measured", statement=f"Most curtailed resource: {top['plant']} ({top['mwh']:.0f} MWh, reason: {top['reason']}).", source="synthetic curtailment events"))
        gm = g["metrics"]
        beats = sum(1 for v in gm.values() if v["beats_best_baseline_mae"])
        f.append(Finding(kind="measured", statement=f"Generation models beat the best baseline on MAE in {beats} of {len(gm)} source/horizon pairs (wind 72 h and thermal 6-72 h do not).", source="holdout evaluation"))
        recs = []
        if c["total_curtailed_mwh"] > 100:
            recs.append("Evaluate battery storage or demand shifting to absorb curtailed wind energy (see storage optimisation).")
        if res[:24].min() < 10:
            recs.append("Review thermal and import availability for the tightest hours; consider pre-positioning reserves.")
        h24 = gm["thermal_h24"]["model"]["mape_pct"]
        return AgentOutput(agent=self.name, status="ok", summary=f"Min 24 h reserve {res[:24].min():.1f}%; {int((un > 1).sum())} unserved hours in 72 h.", findings=f,
                           recommendations=recs, citations=self.knowledge(u, "renewable curtailment transmission limits storage"),
                           confidence=0.7, confidence_definition="Fixed 0.7: mixed model skill across sources; see per-source metrics",
                           uncertainty="Source forecasts carry 80% intervals in /generation; reserve margin assumes scheduled availability is known.",
                           data_origin="forecast", model_version=g["model_version"],
                           limitations=["Hydro and thermal come from a merit-order engine, not from the ML models.", g["note"]], tools_used=u)


class MaintenanceAgent(Agent):
    name = "predictive_maintenance"
    responsibility = "Rank units by 14-day failure risk and explain drivers; flag anomalous telemetry."
    tools = ["get_maintenance_risk", "get_anomalies", "search_knowledge"]
    kb_category = "maintenance"

    def run(self, inp):
        u: list[str] = []
        m = self.call("get_maintenance_risk", u)
        a = self.call("get_anomalies", u)
        top = m["units"][:5]
        mm = m["metrics"]["model"]
        f = [Finding(kind="prediction", statement=f"{x['unit_id']} ({x['plant_name']}, {x['source']}) has predicted 14-day fault probability {x['risk_probability']:.0%}; drivers: {'; '.join(x['drivers']) or 'model-level pattern'}.",
                     source=m["model_version"], value=x["risk_probability"]) for x in top]
        f.append(Finding(kind="measured", statement=f"{sum(1 for x in m['units'] if x['flagged'])} of {len(m['units'])} units exceed the decision threshold ({m['decision_threshold']:.2f}).", source=m["model_version"]))
        f.append(Finding(kind="measured", statement=f"Holdout performance: precision {mm['precision']:.2f}, recall {mm['recall']:.2f}, F1 {mm['f1']:.2f}, AUC {mm.get('roc_auc', float('nan')):.2f} (vibration-threshold baseline F1 {m['metrics']['baseline_vibration_threshold_p90']['f1']:.2f}).", source="holdout evaluation"))
        f.append(Finding(kind="measured", statement=f"{len(a['events'])} anomalous telemetry hours flagged in the last {a['window_hours']} h.", source=a["model_version"]))
        recs = [f"Prioritise inspection of {x['unit_id']} ({x['plant_name']}): {x['risk_probability']:.0%} risk, {'; '.join(x['drivers'][:2]) or 'elevated model score'}." for x in top[:3]]
        return AgentOutput(agent=self.name, status="ok", summary=f"Top risk: {top[0]['unit_id']} at {top[0]['risk_probability']:.0%}.", findings=f, recommendations=recs,
                           citations=self.knowledge(u, "turbine vibration bearing temperature condition monitoring"),
                           confidence=float(mm["precision"]), confidence_definition="Holdout precision at the decision threshold (share of flagged units that faulted within 14 days)",
                           uncertainty=f"Expect roughly {1 - mm['precision']:.0%} of flagged units to be false alarms.", data_origin="synthetic+model",
                           model_version=m["model_version"], limitations=m["limitations"] + ["Risk is for prioritising inspections, not a diagnosis."], tools_used=u)


class GridAgent(Agent):
    name = "grid_intelligence"
    responsibility = "Detect transmission bottlenecks and contingency violations on an explicit network model."
    tools = ["get_grid_status", "run_n1_screen", "search_knowledge"]
    kb_category = "grid"

    def run(self, inp):
        u: list[str] = []
        g = self.call("get_grid_status", u)
        n1 = self.call("run_n1_screen", u)
        bn = g["bottlenecks"]
        over = [l for l in g["lines"] if l["hours_over_100pct"] > 0]
        f = [Finding(kind="prediction", statement=f"{l['line_id']} peaks at {l['max_loading_pct_72h']:.0f}% of assumed capacity over 72 h ({l['hours_over_80pct']} h above 80%).", source="DC power-flow (illustrative)", value=l["max_loading_pct_72h"]) for l in g["lines"] if l["max_loading_pct_72h"] > 60][:5]
        f.append(Finding(kind="prediction", statement=f"N-1 screening at {n1['time']}: {n1['contingencies_with_violations']} of {n1['n_contingencies']} single-line outages cause overloads or islanding.", source="DC power-flow N-1"))
        for v in [x for x in n1["violations"] if x["overloads"]][:3]:
            f.append(Finding(kind="prediction", statement=f"Loss of {v['outage']} overloads {v['overloads'][0]['line_id']} to {v['overloads'][0]['loading_pct']:.0f}%.", source="DC power-flow N-1"))
        isl = [x["outage"] for x in n1["violations"] if x["islanding"]]
        if isl:
            f.append(Finding(kind="assumption", statement="Outages that island a bus (" + ", ".join(isl[:5]) + ") reflect radial spurs in the simplified network, not necessarily real redundancy gaps.", source="network model"))
        recs = []
        if over:
            recs.append(f"Review operating limits and dispatch on {over[0]['line_id']}; the model shows overload in {over[0]['hours_over_100pct']} h.")
        for v in [x for x in n1["violations"] if x["overloads"]][:1]:
            recs.append(f"Prepare a remedial action plan for the {v['outage']} contingency (overload of {v['overloads'][0]['line_id']}).")
        return AgentOutput(agent=self.name, status="ok", summary=f"{len(bn)} bottleneck(s) >70%; {n1['contingencies_with_violations']} N-1 violations.", findings=f, recommendations=recs,
                           citations=self.knowledge(u, "DC power flow approximation limits congestion"), confidence=0.5,
                           confidence_definition="Fixed 0.5: illustrative network with assumed capacities, not validated",
                           uncertainty="Capacities and reactances are assumptions; treat loading percentages as relative indicators.", data_origin="forecast+synthetic",
                           model_version="dc-network-1.0", limitations=[g["label"]], tools_used=u)


class StorageAgent(Agent):
    name = "energy_storage_optimization"
    responsibility = "Optimise simulated battery schedules and compare storage sizing options."
    tools = ["optimize_storage", "get_curtailment", "search_knowledge"]
    kb_category = "renewables"

    def run(self, inp):
        u: list[str] = []
        p = inp.params
        pw, en = float(p.get("power_mw", 100)), float(p.get("energy_mwh", 400))
        r = self.call("optimize_storage", u, power_mw=pw, energy_mwh=en)
        if not r.get("feasible"):
            return AgentOutput(agent=self.name, status="error", summary="Optimiser found no feasible schedule.", error=r.get("status"), tools_used=u)
        opts = []
        for (a, b) in [(50, 200), (100, 400), (200, 800)]:
            x = self.call("optimize_storage", u, power_mw=a, energy_mwh=b)
            opts.append((a, b, x["peak_reduction_mw"], x["energy_discharged_mwh"]))
        f = [Finding(kind="prediction", statement=f"{pw:.0f} MW / {en:.0f} MWh battery lowers the 24 h residual-load peak from {r['peak_before_mw']:.0f} to {r['peak_after_mw']:.0f} MW ({r['peak_reduction_mw']:.0f} MW).", source="LP optimisation (GLOP)", value=r["peak_reduction_mw"]),
             Finding(kind="prediction", statement=f"Schedule discharges {r['energy_discharged_mwh']:.0f} MWh ({r['equivalent_full_cycles']:.2f} cycles); status {r['status']}, validated: {r['schedule_valid']}.", source="LP optimisation"),
             Finding(kind="prediction", statement="Compared sizes: " + "; ".join(f"{a}MW/{b}MWh -> {pr:.0f} MW peak cut" for a, b, pr, _ in opts) + ".", source="LP optimisation"),
             Finding(kind="assumption", statement=f"Round-trip efficiency {r['assumptions']['round_trip_efficiency']:.0%}, reserve {r['assumptions']['reserve_frac']:.0%} SOC, price signal is a synthetic proxy.", source="optimiser inputs")]
        best = max(opts, key=lambda x: x[2] / x[1])
        return AgentOutput(agent=self.name, status="ok", summary=f"Peak cut of {r['peak_reduction_mw']:.0f} MW with {pw:.0f}/{en:.0f}.", findings=f,
                           recommendations=[f"Shortlist the {best[0]} MW / {best[1]} MWh option for detailed siting and cost study (best peak cut per MWh in this comparison)."],
                           citations=self.knowledge(u, "battery storage curtailment peak shaving"), confidence=0.6,
                           confidence_definition="Fixed 0.6: optimal for the stated forecast, but forecast error propagates", uncertainty="Schedule is optimal for the point forecast only; no stochastic optimisation.",
                           data_origin="forecast+synthetic", model_version="lp-glop-1.0", limitations=["No degradation curve; throughput cost proxy only.", "No network siting constraints."], tools_used=u)


class LossAgent(Agent):
    name = "electricity_loss_detection"
    responsibility = "Separate technical from non-technical losses and surface feeders with unusual unaccounted energy."
    tools = ["get_loss_analysis", "search_knowledge"]
    kb_category = "losses"

    def run(self, inp):
        u: list[str] = []
        l = self.call("get_loss_analysis", u)
        regs = sorted(l["regions"], key=lambda r: -r["ntl_pct"])
        t = l["totals"]
        f = [Finding(kind="measured", statement=f"Latest month: input {t['input_mwh']:.0f} MWh, metered {t['metered_mwh']:.0f} MWh, total loss {t['loss_pct']:.1f}% (est. NTL {t['ntl_mwh']:.0f} MWh, technical {t['tech_mwh']:.0f} MWh).", source="synthetic feeder energy balance")]
        f += [Finding(kind="measured", statement=f"{r['region']}: loss {r['loss_pct']:.1f}% (NTL est. {r['ntl_pct']:.1f}%, technical {r['tech_pct']:.1f}%), {int(r['flagged'])} flagged feeders.", source=l["model_version"]) for r in regs[:3]]
        top = l["flagged_feeders"][:3]
        f += [Finding(kind="prediction", statement=f"Feeder {x['feeder_id']} ({x['region']}) flagged: est. NTL {x['ntl_est_pct']:.1%} (~{x['ntl_mwh_est']:.0f} MWh/month).", source=l["model_version"]) for x in top]
        mt = l["metrics"]
        f.append(Finding(kind="measured", statement=f"Held-out F1: hybrid {mt['hybrid_iforest_and_ntl_ge_4pct']['f1']:.2f}, isolation forest alone {mt['isolation_forest']['f1']:.2f}, 5% rule {mt['baseline_ntl_est_ge_5pct']['f1']:.2f}.", source="holdout evaluation (months 9-11)"))
        recs = [f"Schedule field meter audits for feeders {', '.join(x['feeder_id'] for x in top)} before any enforcement action."] if top else []
        return AgentOutput(agent=self.name, status="ok", summary=f"Total loss {t['loss_pct']:.1f}%; highest NTL in {regs[0]['region']} ({regs[0]['ntl_pct']:.1f}%).", findings=f, recommendations=recs,
                           citations=self.knowledge(u, "technical and non-technical losses feeders"), confidence=0.6,
                           confidence_definition="Fixed 0.6: injected losses are easier to detect than real ones", uncertainty="Real-world precision is unknown; metrics come from synthetic injected losses.",
                           data_origin="synthetic+model", model_version=l["model_version"], limitations=[l["caveat"]], tools_used=u)


class ClimateAgent(Agent):
    name = "climate_intelligence"
    responsibility = "Summarise weather and hydrology drivers affecting solar, wind, hydro and demand."
    tools = ["get_climate", "get_overview", "search_knowledge"]
    kb_category = "climate"

    def run(self, inp):
        u: list[str] = []
        c = self.call("get_climate", u)
        o = self.call("get_overview", u)
        wf = c["water_factor_history"]
        trend = wf[-1] - wf[-15]
        fut = {r["region"]: r for r in c["next_24h_mean_synthetic_forecast"]}
        windiest = max(fut.values(), key=lambda r: r["wind_ms"])
        sunniest = max(fut.values(), key=lambda r: r["ghi_wm2"])
        f = [Finding(kind="measured", statement=f"Water factor is {wf[-1]:.2f} ({'rising' if trend > 0.02 else 'falling' if trend < -0.02 else 'stable'} over 14 days, {trend:+.2f}).", source="synthetic rainfall-derived index"),
             Finding(kind="prediction", statement=f"Next-24h synthetic forecast: windiest region {windiest['region']} ({windiest['wind_ms']:.1f} m/s mean), highest mean irradiance {sunniest['region']} ({sunniest['ghi_wm2']:.0f} W/m2).", source="synthetic weather forecast"),
             Finding(kind="assumption", statement="Hydro availability scales with the water factor (0.45-0.95); this is a modelling convenience, not a reservoir measurement.", source="synthetic generator")]
        return AgentOutput(agent=self.name, status="ok", summary=f"Water factor {wf[-1]:.2f}; wind strongest in {windiest['region']}.", findings=f,
                           recommendations=["Connect Open-Meteo / NASA POWER ingestion to replace synthetic weather before relying on climate outputs."],
                           citations=self.knowledge(u, "hydropower rainfall variability reservoir"), confidence=0.4,
                           confidence_definition="Fixed 0.4: weather is synthetic", uncertainty="No real meteorological data is connected.", data_origin="synthetic", model_version="synthetic-weather-1.0",
                           limitations=[c["note"]], tools_used=u)


SPECIALISTS = [DemandAgent, GenerationAgent, MaintenanceAgent, GridAgent, StorageAgent, LossAgent, ClimateAgent]


class ChiefAgent(Agent):
    name = "chief_energy_intelligence"
    responsibility = "Coordinate specialist agents, resolve conflicting evidence, identify missing information and issue a consolidated assessment."
    tools = ["run_scenario", "get_overview", "search_knowledge"]
    min_role = "viewer"

    def __init__(self, svc, tools, registry: "AgentRegistry"):
        super().__init__(svc, tools)
        self.registry = registry

    def assess(self, question: str | None, names: list[str] | None, actor: str, role: str, audit: AuditLog | None) -> dict:
        t0 = time.perf_counter()
        names = names or [c.name for c in SPECIALISTS]
        outs: dict[str, AgentOutput] = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {n: ex.submit(self.registry.run, n, AgentInput(question=question), actor, role, audit) for n in names}
            for n, fu in futs.items():
                outs[n] = fu.result()
        conflicts, missing, recs, used = [], [], [], []
        b = self.svc.baseline
        res24 = float(b["reserve_margin_pct"][:24].min())
        loading = float(b["loading"][:24].max() * 100)
        unserved = int((b["unserved"][:24] > 1).sum())
        if res24 >= 10 and loading > 100:
            conflicts.append({"between": ["generation_intelligence", "grid_intelligence"], "resolution": "National capacity is adequate (reserve "
                              f"{res24:.0f}%) but at least one line exceeds assumed capacity ({loading:.0f}%). Treated as a network constraint, not a capacity shortage."})
        elif res24 < 10 and loading < 80:
            conflicts.append({"between": ["generation_intelligence", "grid_intelligence"], "resolution": f"Reserve is thin ({res24:.0f}%) while the network is not congested ({loading:.0f}%); treated as a generation-adequacy issue."})
        # maintenance x generation coupling: run a loss-of-plant scenario for the riskiest unit
        extra = None
        mr = self.svc.maint_risk
        if len(mr):
            top = mr.iloc[0]
            pl = self.svc.plants.set_index("plant_id").loc[top.plant_id]
            frac = float((self.svc.world.units.set_index("unit_id").loc[top.unit_id, "rated_mw"]) / pl.capacity_mw)
            sc = self.call("run_scenario", used, name=f"Loss of {top.unit_id} (risk {top.risk_probability:.0%})", horizon_hours=24,
                           plant_outages=[{"plant_id": top.plant_id, "start_hour": 0, "duration_h": 24, "fraction": frac}])
            extra = {"unit": top.unit_id, "scenario": {"unserved_mwh_delta": sc["delta_vs_baseline"]["unserved_energy_mwh"], "min_reserve_pct": sc["scenario"]["min_reserve_margin_pct"]}}
            if sc["delta_vs_baseline"]["unserved_energy_mwh"] > 1:
                conflicts.append({"between": ["predictive_maintenance", "generation_intelligence"], "resolution": f"Loss of {top.unit_id} would add {sc['delta_vs_baseline']['unserved_energy_mwh']:.0f} MWh unserved energy; maintenance priority raised."})
        for n, o in outs.items():
            if o.status != "ok":
                missing.append(f"{n}: {o.status} ({o.error or 'no detail'})")
            recs += o.recommendations
        if not any(o.citations for o in outs.values()):
            missing.append("No engineering-knowledge citations retrieved.")
        missing += ["No live SCADA, EMS or smart-meter feeds are connected; all history is synthetic.",
                    "No utility-supplied equipment manuals or regulator reports are ingested; the knowledge base holds only author-written reference notes.",
                    "Weather inputs are synthetic (Open-Meteo / NASA POWER not enabled)."]
        level = "high" if (unserved > 0 or res24 < 5 or loading > 100) else "elevated" if (res24 < 10 or loading > 85) else "moderate" if self.svc.maint_risk.flagged.sum() or len(self.svc.anomalies) else "low"
        mk = lambda s: Finding(kind="measured", statement=s, source="chief rollup")
        findings = [mk(f"Lowest 24 h reserve margin {res24:.1f}%, peak line loading {loading:.0f}% (DC model), {unserved} unserved hours."),
                    Finding(kind="assumption", statement="Risk level rule: high if unserved hours>0, reserve<5% or any line>100%; elevated if reserve<10% or line>85%; moderate if any unit flagged or telemetry anomaly; else low.", source="chief rules")]
        out = {"agent": self.name, "overall_risk_level": level, "question": question, "agents_run": list(outs), "conflicts_resolved": conflicts,
               "missing_information": missing, "coupled_scenario": extra, "findings": [f.model_dump() for f in findings],
               "recommendations": recs[:8], "specialists": {n: o.model_dump() for n, o in outs.items()},
               "timestamp": self.svc._now(), "model_versions": self.svc.model_versions, "approval_required": True,
               "limitations": ["Advisory only; no command path to infrastructure exists.", "All metrics are from synthetic holdout data."],
               "duration_ms": round((time.perf_counter() - t0) * 1000, 1), "tools_used": used}
        if audit:
            audit.append("chief_assessment", actor, role, {"level": level, "agents": list(outs), "conflicts": len(conflicts), "question": question})
        return out


class AgentRegistry:
    def __init__(self, svc: PowerIQ):
        self.svc = svc
        self.tools = build_tools(svc)
        self.agents: dict[str, Agent] = {c.name: c(svc, self.tools) for c in SPECIALISTS}
        self.chief = ChiefAgent(svc, self.tools, self)
        self.stats: dict[str, dict] = {n: {"runs": 0, "errors": 0, "last_run": None, "last_duration_ms": None} for n in self.agents}
        self._pool = ThreadPoolExecutor(max_workers=8)

    def run(self, name: str, inp: AgentInput, actor: str = "system", role: str = "viewer", audit: AuditLog | None = None) -> AgentOutput:
        if name not in self.agents:
            raise KeyError(name)
        ag = self.agents[name]
        if ROLE_LEVEL[role] < ROLE_LEVEL[ag.min_role]:
            raise PermissionError(f"role {role} cannot run {name}")
        t0 = time.perf_counter()
        fut = self._pool.submit(ag.run, inp)
        try:
            out = fut.result(timeout=get_settings().agent_timeout_s)
        except FutTimeout:
            out = AgentOutput(agent=name, status="error", summary="Agent timed out.", error="timeout")
        except Exception as e:  # error handling: never propagate agent internals to the caller
            out = AgentOutput(agent=name, status="error", summary="Agent failed.", error=f"{type(e).__name__}: {e}")
        out.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        out.timestamp = self.svc._now()
        s = self.stats[name]
        s["runs"] += 1
        s["errors"] += int(out.status == "error")
        s["last_run"], s["last_duration_ms"] = out.timestamp, out.duration_ms
        if audit:
            audit.append("agent_run", actor, role, {"agent": name, "status": out.status, "duration_ms": out.duration_ms,
                                                    "tools": out.tools_used, "error": out.error, "question": inp.question})
        return out

    def status(self) -> list[dict]:
        rows = [{**a.info(), **self.stats[n]} for n, a in self.agents.items()]
        rows.append({**self.chief.info(), "runs": None, "errors": None, "note": "Coordinates all specialists via /copilot/chat and /agents/chief/assess"})
        return rows
