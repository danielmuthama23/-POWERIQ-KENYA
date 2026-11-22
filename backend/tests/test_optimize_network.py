import numpy as np
import pytest
from app.network import DCNetwork
from app.optimize import BatterySpec, optimize_battery


def _load():
    h = np.arange(24)
    return 800 + 300 * np.exp(-0.5 * ((h - 19.5) / 2) ** 2)


def test_battery_reduces_peak_and_is_valid():
    r = optimize_battery(_load(), BatterySpec(100, 400))
    assert r["feasible"] and r["schedule_valid"]
    assert r["peak_after_mw"] < r["peak_before_mw"]
    assert min(r["soc_mwh"]) >= 0.15 * 400 - 1e-6 and max(r["soc_mwh"]) <= 400 + 1e-6
    assert r["soc_mwh"][-1] >= 0.5 * 400 - 1e-6
    assert max(r["charge_mw"]) <= 100 + 1e-6 and max(r["discharge_mw"]) <= 100 + 1e-6


def test_battery_energy_conservation_with_losses():
    r = optimize_battery(_load(), BatterySpec(100, 400, round_trip_eff=0.88))
    assert r["energy_charged_mwh"] * 0.88 >= r["energy_discharged_mwh"] - 1e-3


def test_larger_battery_never_worse():
    a = optimize_battery(_load(), BatterySpec(50, 200))["peak_after_mw"]
    b = optimize_battery(_load(), BatterySpec(200, 800))["peak_after_mw"]
    assert b <= a + 1e-6


@pytest.mark.parametrize("spec", [BatterySpec(0, 100), BatterySpec(100, -5), BatterySpec(10, 10, reserve_frac=1.2)])
def test_invalid_battery_specs(spec):
    with pytest.raises(ValueError):
        optimize_battery(_load(), spec)


def test_dc_flow_conserves_power(svc):
    net = svc.net
    inj = np.random.default_rng(0).normal(0, 50, (5, len(net.buses)))
    inj -= inj.mean(axis=1, keepdims=True)
    f, _ = net.flows(inj)
    # net outflow at each bus equals its injection
    out = f @ net.A
    assert np.allclose(out, inj, atol=1e-6)


def test_disconnected_network_raises(svc):
    keep = svc.net.lines.drop(index=[svc.net.lines.index[svc.net.lines.line_id == "LESSOS-ELDORET"][0]])
    with pytest.raises(np.linalg.LinAlgError):
        DCNetwork(svc.world.substations, keep.reset_index(drop=True))


def test_scenario_monotonic(svc):
    base = svc.scenario({"generation_reductions": []})
    stress = svc.scenario({"generation_reductions": [{"source": "geothermal", "pct": 60}, {"source": "hydro", "pct": 60}]})
    assert stress["scenario"]["unserved_energy_mwh"] > base["scenario"]["unserved_energy_mwh"]
    assert stress["delta_vs_baseline"]["min_reserve_margin_pct"] < 0
    assert "NOT a validated" in stress["label"]


def test_battery_helps_in_stress_scenario(svc):
    req = {"generation_reductions": [{"source": "hydro", "pct": 60}, {"source": "geothermal", "pct": 25}], "demand_increase_pct": 10}
    a = svc.scenario(req)
    b = svc.scenario({**req, "battery": {"power_mw": 200, "energy_mwh": 800}})
    assert b["scenario"]["unserved_energy_mwh"] <= a["scenario"]["unserved_energy_mwh"] + 1e-6


def test_scenario_rejects_unknown_ids(svc):
    with pytest.raises(ValueError):
        svc.scenario({"plant_outages": [{"plant_id": "NOPE"}]})
    with pytest.raises(ValueError):
        svc.scenario({"line_derates": [{"line_id": "A-B", "derate_pct": 50}]})


def test_map_cells_contract(svc):
    m = svc.map_cells(5)
    assert len(m["features"]) == len(svc.cells)
    keys = set(svc.METRICS)
    p = m["features"][0]["properties"]
    assert keys <= set(p)
    assert all(0 <= f["properties"]["composite_risk"] <= 100 for f in m["features"])
    tot = sum(f["properties"]["demand_mw"] for f in m["features"])
    assert abs(tot - svc.baseline["demand"][5]) < 0.5
