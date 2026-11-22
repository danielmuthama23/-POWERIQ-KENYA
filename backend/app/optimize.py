"""Battery energy storage optimisation (linear programme, OR-Tools GLOP).

Minimises: peak_weight * peak_net_load + energy_cost + degradation_cost
Subject to: power limits, SOC bounds incl. reserve, round-trip efficiency, end-of-horizon SOC >= start SOC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp


@dataclass
class BatterySpec:
    power_mw: float
    energy_mwh: float
    round_trip_eff: float = 0.88
    soc_start_frac: float = 0.50
    reserve_frac: float = 0.15           # state of charge never allowed below this fraction
    degradation_cost_per_mwh: float = 4.0  # throughput cost proxy (USD/MWh), assumption
    max_daily_cycles: float = 1.5


def optimize_battery(net_load_mw: np.ndarray, spec: BatterySpec, price_per_mwh: np.ndarray | None = None,
                     free_charge_mw: np.ndarray | None = None, peak_weight: float = 250.0, dt_h: float = 1.0) -> dict:
    """net_load_mw: load the battery can shave (demand minus must-run/variable supply)."""
    T = len(net_load_mw)
    if spec.power_mw <= 0 or spec.energy_mwh <= 0:
        raise ValueError("Battery power and energy must be positive.")
    if not (0 <= spec.reserve_frac < 1) or not (0 < spec.round_trip_eff <= 1):
        raise ValueError("Invalid reserve fraction or efficiency.")
    price = np.full(T, 60.0) if price_per_mwh is None else np.asarray(price_per_mwh, float)
    eta = float(np.sqrt(spec.round_trip_eff))
    s = pywraplp.Solver.CreateSolver("GLOP")
    if s is None:
        raise RuntimeError("GLOP solver unavailable")
    P, E = spec.power_mw, spec.energy_mwh
    ch = [s.NumVar(0, P, f"ch{t}") for t in range(T)]
    dis = [s.NumVar(0, P, f"dis{t}") for t in range(T)]
    soc = [s.NumVar(spec.reserve_frac * E, E, f"soc{t}") for t in range(T)]
    peak = s.NumVar(0, s.infinity(), "peak")
    soc0 = spec.soc_start_frac * E
    for t in range(T):
        prev = soc0 if t == 0 else soc[t - 1]
        s.Add(soc[t] == prev + eta * ch[t] * dt_h - dis[t] * dt_h / eta)
        s.Add(peak >= float(net_load_mw[t]) + ch[t] - dis[t])
    s.Add(soc[T - 1] >= soc0)
    s.Add(sum((ch[t] + dis[t]) * dt_h for t in range(T)) <= 2 * E * spec.max_daily_cycles * (T / 24.0))
    free = np.zeros(T) if free_charge_mw is None else np.asarray(free_charge_mw, float)
    obj = peak_weight * peak
    for t in range(T):
        # energy that would otherwise be curtailed is charged at zero cost
        charge_cost = 0.0 if free[t] >= P * 0.5 else price[t]
        obj += (charge_cost * ch[t] - price[t] * dis[t]) * dt_h * 0.02 + spec.degradation_cost_per_mwh * (ch[t] + dis[t]) * dt_h * 0.02
    s.Minimize(obj)
    status = s.Solve()
    names = {pywraplp.Solver.OPTIMAL: "optimal", pywraplp.Solver.FEASIBLE: "feasible",
             pywraplp.Solver.INFEASIBLE: "infeasible", pywraplp.Solver.UNBOUNDED: "unbounded"}
    st = names.get(status, f"status_{status}")
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {"status": st, "feasible": False}
    c = np.array([v.solution_value() for v in ch])
    d = np.array([v.solution_value() for v in dis])
    so = np.array([v.solution_value() for v in soc])
    after = net_load_mw + c - d
    # independent validation of the returned schedule
    tol = 1e-4
    sim = soc0
    ok = True
    for t in range(T):
        sim = sim + eta * c[t] * dt_h - d[t] * dt_h / eta
        ok &= (spec.reserve_frac * E - tol <= sim <= E + tol)
    ok &= bool(np.all(c <= P + tol) and np.all(d <= P + tol))
    throughput = float((c + d).sum() * dt_h)
    return {
        "status": st, "feasible": True, "schedule_valid": bool(ok),
        "charge_mw": c.tolist(), "discharge_mw": d.tolist(), "soc_mwh": so.tolist(),
        "net_load_before_mw": net_load_mw.tolist(), "net_load_after_mw": after.tolist(),
        "peak_before_mw": float(net_load_mw.max()), "peak_after_mw": float(after.max()),
        "peak_reduction_mw": float(net_load_mw.max() - after.max()),
        "energy_discharged_mwh": float(d.sum() * dt_h), "energy_charged_mwh": float(c.sum() * dt_h),
        "equivalent_full_cycles": float(d.sum() * dt_h / E), "throughput_mwh": throughput,
        "degradation_cost_usd": float(throughput * spec.degradation_cost_per_mwh),
        "assumptions": {"round_trip_efficiency": spec.round_trip_eff, "reserve_frac": spec.reserve_frac,
                        "degradation_cost_per_mwh_usd": spec.degradation_cost_per_mwh,
                        "price_signal": "synthetic proxy (USD/MWh), not a tariff"},
    }
