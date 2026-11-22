import numpy as np
from app import ml, synthetic


def test_world_is_reproducible_and_labelled():
    a, b = synthetic.make_world(7), synthetic.make_world(7)
    assert np.allclose(a.demand.values, b.demand.values)
    assert a.provenance["data_origin"] == "synthetic" and a.provenance["seed"] == 7
    for df in (a.plants, a.cells, a.units, a.telemetry, a.feeder_monthly):
        assert (df.data_origin == "synthetic").all()
    assert len(a.times) == a.n_hist + synthetic.HORIZON_MAX


def test_physical_consistency(svc):
    w = svc.world
    sys = w.system.iloc[: w.n_hist]
    assert (w.plant_gen.values >= -1e-9).all()
    assert (w.plant_gen <= w.plants.capacity_mw.values + 1e-6).all().all()
    assert (sys.unserved_mw >= 0).all()
    assert abs((sys.supplied_mw + sys.unserved_mw - sys.required_gen_mw).abs().max()) < 1e-6


def test_demand_model_beats_baseline_on_holdout(svc):
    for h in ml.HORIZONS:
        m = svc.fb.metrics["demand"][f"h{h}"]["national"]
        assert m["model"]["mape_pct"] < m["baseline_same_hour_last_week"]["mape_pct"], h
        assert m["model"]["mae"] > 0 and m["model"]["rmse"] >= m["model"]["mae"]


def test_forecast_error_grows_with_horizon(svc):
    mapes = [svc.fb.metrics["demand"][f"h{h}"]["national"]["model"]["mape_pct"] for h in ml.HORIZONS]
    assert mapes[0] < mapes[-1]


def test_interval_coverage_reasonable(svc):
    for h in ml.HORIZONS:
        assert 0.75 <= svc.fb.metrics["demand"][f"h{h}"]["interval_80_empirical_coverage"] <= 0.85


def test_forecast_shapes_and_bounds(svc):
    assert svc.dem_fc.shape == (72, 10) and (svc.dem_fc.values > 0).all()
    caps = svc.world.plants.groupby("source").capacity_mw.sum()
    for s in ml.SOURCES_MODELLED:
        v = svc.gen_fc[s].values
        assert (v >= 0).all() and (v <= caps[s] * 1.06).all()
        assert (svc.gen_lo[s].values <= v + 1e-6).all() and (svc.gen_hi[s].values >= v - 1e-6).all()


def test_generation_metrics_report_baseline_comparison(svc):
    gm = svc.fb.metrics["generation"]
    assert len(gm) == 20
    assert all("beats_best_baseline_mae" in v for v in gm.values())
    assert gm["solar_h24"]["beats_best_baseline_mae"] and gm["geothermal_h24"]["beats_best_baseline_mae"]


def test_maintenance_metrics_and_ranking(svc):
    m = svc.mb.metrics
    assert m["model"]["roc_auc"] > 0.7
    assert m["model"]["f1"] > m["baseline_vibration_threshold_p90"]["f1"]
    cm = m["model"]["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == m["test_rows"]
    assert svc.maint_risk.risk_probability.is_monotonic_decreasing


def test_anomaly_detection(svc):
    m = svc.ab.metrics["isolation_forest"]
    assert m["f1"] > 0.5 and m["roc_auc"] > 0.8


def test_loss_detection(svc):
    m = svc.lb.metrics
    assert m["hybrid_iforest_and_ntl_ge_4pct"]["recall"] > 0.8
    assert (svc.lb.latest.ntl_mwh_est >= 0).all()
