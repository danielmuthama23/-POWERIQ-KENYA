"""Independently testable ML pipelines for POWERIQ Kenya.

Every pipeline: builds features -> time-based split -> trains -> evaluates on a held-out window
against explicit baselines -> reports the *measured* metrics. Nothing here hard-codes a metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from . import geo
from .synthetic import HOLIDAYS, HORIZON_MAX, World

HORIZONS = [1, 6, 24, 72]
TEST_HOURS = 24 * 28
SOURCES_MODELLED = ["geothermal", "hydro", "wind", "solar", "thermal"]
MODEL_VERSION = "poweriq-ml-1.0.0"


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def reg_metrics(y: np.ndarray, p: np.ndarray, mape_floor: float | None = None) -> dict[str, float | None]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    out: dict[str, float | None] = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
                                   "n": int(len(y))}
    floor = mape_floor if mape_floor is not None else max(1e-9, 0.05 * np.max(np.abs(y)))
    m = np.abs(y) > floor
    out["mape_pct"] = float(np.mean(np.abs(err[m]) / np.abs(y[m])) * 100) if m.any() else None
    return out


def cls_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, Any]:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {"precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
           "n": int(len(y)), "positives": int(y.sum())}
    if score is not None and len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, score))
    return out


# ------------------------------------------------------------------------------------------
# shared feature building for demand and generation forecasting
# ------------------------------------------------------------------------------------------
def _calendar(times: pd.DatetimeIndex) -> pd.DataFrame:
    doy = times.dayofyear.values
    return pd.DataFrame({
        "hour": times.hour.values, "dow": times.dayofweek.values,
        "holiday": np.isin(times.strftime("%Y-%m-%d"), list(HOLIDAYS)).astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=times)


def _fc_noise(h: int, n: int, scale: float, seed: int) -> np.ndarray:
    """Synthetic NWP error grows with sqrt(lead time); deterministic per (h, seed)."""
    rng = np.random.default_rng(seed + 1000 * h)
    return rng.normal(0, scale * (0.25 + 0.12 * math.sqrt(h)), n)


def _lag(arr: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    j = idx - k
    out = np.full(len(idx), np.nan)
    ok = j >= 0
    out[ok] = arr[j[ok]]
    return out


@dataclass
class ForecastBundle:
    version: str
    demand_models: dict[int, Any] = field(default_factory=dict)
    gen_models: dict[tuple[str, int], Any] = field(default_factory=dict)
    demand_scale: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    demand_rel_err_q: dict[int, tuple[float, float]] = field(default_factory=dict)   # (q10, q90) of relative error
    gen_err_q: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)  # absolute MW quantiles
    feature_names_demand: list[str] = field(default_factory=list)
    feature_names_gen: list[str] = field(default_factory=list)


def _demand_features(world: World, region_idx: int, T: np.ndarray, h: int, scale: float, seed: int) -> pd.DataFrame:
    rid = geo.REGION_IDS[region_idx]
    y = world.demand[rid].values
    w = world.weather[world.weather.region == rid].set_index("time").reindex(world.times)
    cal = _calendar(world.times[T])
    k_day = 24 * math.ceil(h / 24)
    f = pd.DataFrame({
        "region_code": region_idx,
        "y_origin": _lag(y, T, h) / scale,
        "y_lag_day": _lag(y, T, k_day) / scale,
        "y_lag_week": _lag(y, T, 168) / scale,
        "y_lag_week_prev": _lag(y, T, 168 + 24) / scale,
        "roll24_origin": np.array([np.nanmean(y[max(0, t - h - 23): t - h + 1]) if t - h >= 0 else np.nan for t in T]) / scale,
        "temp_fc": w.temp_c.values[T] + _fc_noise(h, len(T), 1.0, seed + region_idx),
    }, index=world.times[T])
    return pd.concat([f, cal], axis=1)


def train_demand(world: World, seed: int = 42) -> tuple[dict[int, Any], dict[str, float], dict, dict, list[str]]:
    n_hist = world.n_hist
    split = n_hist - TEST_HOURS
    scale = {rid: float(world.demand[rid].iloc[:split].mean()) for rid in geo.REGION_IDS}
    models: dict[int, Any] = {}
    metrics: dict[str, Any] = {}
    qerr: dict[int, tuple[float, float]] = {}
    names: list[str] = []
    for h in HORIZONS:
        Xtr, ytr, Xte, yte, meta_te = [], [], [], [], []
        for ri, rid in enumerate(geo.REGION_IDS):
            T_tr = np.arange(169 + 24, split)
            T_te = np.arange(split, n_hist)
            for T, X, Y in ((T_tr, Xtr, ytr), (T_te, Xte, yte)):
                f = _demand_features(world, ri, T, h, scale[rid], seed)
                X.append(f)
                Y.append(world.demand[rid].values[T] / scale[rid])
            meta_te.append(pd.DataFrame({"region": rid, "time": world.times[T_te], "scale": scale[rid]}))
        Xtr, ytr = pd.concat(Xtr), np.concatenate(ytr)
        Xte, yte = pd.concat(Xte), np.concatenate(yte)
        meta = pd.concat(meta_te, ignore_index=True)
        names = list(Xtr.columns)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=seed,
                              verbose=-1, n_jobs=2)
        m.fit(Xtr, ytr, categorical_feature=["region_code"])
        pred = m.predict(Xte)
        sc = meta["scale"].values
        y_mw, p_mw = yte * sc, pred * sc
        base_week = Xte["y_lag_week"].values * sc
        base_day = Xte["y_lag_day"].values * sc
        meta = meta.assign(y=y_mw, p=p_mw, bw=base_week, bd=base_day)
        nat = meta.groupby("time")[["y", "p", "bw", "bd"]].sum()
        # regional metrics
        reg = reg_metrics(meta.y.values, meta.p.values)
        metrics[f"h{h}"] = {
            "horizon_h": h,
            "national": {"model": reg_metrics(nat.y.values, nat.p.values),
                         "baseline_same_hour_last_week": reg_metrics(nat.y.values, nat.bw.values),
                         "baseline_same_hour_last_available_day": reg_metrics(nat.y.values, nat.bd.values)},
            "regional_pooled": {"model": reg,
                                "baseline_same_hour_last_week": reg_metrics(meta.y.values, meta.bw.values)},
            "peak_hour_national_mae_mw": float(np.abs(nat.p - nat.y)[nat.y >= nat.y.quantile(0.9)].mean()),
            "test_window": [str(world.times[split]), str(world.times[n_hist - 1])],
        }
        rel = ((nat.p - nat.y) / nat.y).values
        qerr[h] = (float(np.quantile(rel, 0.10)), float(np.quantile(rel, 0.90)))
        metrics[f"h{h}"]["interval_80_empirical_coverage"] = float(np.mean((rel >= qerr[h][0]) & (rel <= qerr[h][1])))
        # refit on full history for operational use
        T_all = np.arange(169 + 24, n_hist)
        Xall = pd.concat([_demand_features(world, ri, T_all, h, scale[rid], seed) for ri, rid in enumerate(geo.REGION_IDS)])
        yall = np.concatenate([world.demand[rid].values[T_all] / scale[rid] for rid in geo.REGION_IDS])
        mf = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=seed,
                               verbose=-1, n_jobs=2)
        mf.fit(Xall, yall, categorical_feature=["region_code"])
        models[h] = mf
    return models, scale, metrics, qerr, names


def lead_to_horizon(lead: int) -> int:
    for h in HORIZONS:
        if lead <= h:
            return h
    return HORIZONS[-1]


def forecast_demand(world: World, bundle: ForecastBundle, n_hours: int = HORIZON_MAX, seed: int = 42) -> pd.DataFrame:
    """Regional demand forecast for the next n_hours after the last observed hour."""
    n_hist = world.n_hist
    out = {}
    leads = np.arange(1, n_hours + 1)
    T = n_hist - 1 + leads
    for ri, rid in enumerate(geo.REGION_IDS):
        pred = np.zeros(n_hours)
        for h in HORIZONS:
            sel = np.array([lead_to_horizon(int(l)) == h for l in leads])
            if not sel.any():
                continue
            f = _demand_features(world, ri, T[sel], h, bundle.demand_scale[rid], seed)
            pred[sel] = bundle.demand_models[h].predict(f[bundle.feature_names_demand]) * bundle.demand_scale[rid]
        out[rid] = pred
    df = pd.DataFrame(out, index=world.times[T])
    df.index.name = "time"
    return df


# ------------------------------------------------------------------------------------------
# generation forecasting
# ------------------------------------------------------------------------------------------
def _source_series(world: World, src: str) -> tuple[np.ndarray, float]:
    ids = world.plants.plant_id[world.plants.source == src].tolist()
    cap = float(world.plants.capacity_mw[world.plants.source == src].sum())
    return world.plant_gen[ids].sum(axis=1).values, cap


def _src_weather(world: World, src: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = world.plants[world.plants.source == src]
    if ids.empty:
        ids = world.plants
    wts = ids.groupby("region").capacity_mw.sum()
    wts = wts / wts.sum()
    wide = {v: world.weather.pivot(index="time", columns="region", values=v).reindex(world.times)
            for v in ("wind_ms", "ghi_wm2", "cloud")}
    wind = sum(wide["wind_ms"][r].values * w for r, w in wts.items())
    ghi = sum(wide["ghi_wm2"][r].values * w for r, w in wts.items())
    cloud = sum(wide["cloud"][r].values * w for r, w in wts.items())
    return wind, ghi, cloud


def _gen_features(world: World, src: str, T: np.ndarray, h: int, seed: int) -> pd.DataFrame:
    y, cap = _source_series(world, src)
    wind, ghi, cloud = _src_weather(world, src)
    dem = world.system.total_demand_mw.values
    ids = world.plants.plant_id[world.plants.source == src].tolist()
    cap_w = world.plants.set_index("plant_id").capacity_mw[ids]
    avail = (world.plant_avail[ids] * cap_w.values).sum(axis=1).values / cap
    k_day = 24 * math.ceil(h / 24)
    cal = _calendar(world.times[T])
    sd = seed + sum(ord(c) for c in src)
    roll24 = np.array([np.nanmean(y[max(0, t - h - 23): t - h + 1]) if t - h >= 0 else np.nan for t in T]) / cap
    f = pd.DataFrame({
        "y_origin": _lag(y, T, h) / cap, "y_lag_day": _lag(y, T, k_day) / cap, "y_lag_week": _lag(y, T, 168) / cap,
        "wind_fc": wind[T] + _fc_noise(h, len(T), 1.5, sd), "ghi_fc": ghi[T] * (1 + _fc_noise(h, len(T), 0.12, sd + 7)),
        "cloud_fc": cloud[T] + _fc_noise(h, len(T), 0.15, sd + 13),
        "dem_origin": _lag(dem, T, h) / 2000.0, "dem_lag_day": _lag(dem, T, k_day) / 2000.0,
        "dem_lag_week": _lag(dem, T, 168) / 2000.0, "avail_scheduled": avail[T],
        "water_factor_origin": _lag(world.system.water_factor.values, T, h),
        "y_roll24_origin": roll24,
        # system demand forecast input (assumption: an upstream demand forecast with ~1-3% error is available)
        "dem_fc": (dem[T] * (1 + _fc_noise(h, len(T), 0.012, sd + 29))) / 2000.0,
        "wind_obs_origin": _lag(wind, T, h), "wind_obs_lag_day": _lag(wind, T, k_day),
        "ghi_obs_lag_day": _lag(ghi, T, k_day),
    }, index=world.times[T])
    return pd.concat([f, cal], axis=1)


def train_generation(world: World, seed: int = 42):
    n_hist = world.n_hist
    split = n_hist - TEST_HOURS
    models, metrics, qerr = {}, {}, {}
    names: list[str] = []
    for src in SOURCES_MODELLED:
        y, cap = _source_series(world, src)
        for h in HORIZONS:
            T_tr = np.arange(169, split)
            T_te = np.arange(split, n_hist)
            Xtr, Xte = _gen_features(world, src, T_tr, h, seed), _gen_features(world, src, T_te, h, seed)
            names = list(Xtr.columns)
            # zero-inflated sources (wind, thermal): L1 objective targets the median, which is what MAE rewards
            obj = "regression_l1" if src in ("wind", "thermal") else "regression"
            kw = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=25, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1, n_jobs=2, objective=obj)
            m = lgb.LGBMRegressor(**kw).fit(Xtr, y[T_tr] / cap)
            pred = np.clip(m.predict(Xte), 0, 1.05) * cap
            yt = y[T_te]
            pers = Xte.y_origin.values * cap
            lagd = Xte.y_lag_day.values * cap
            floor = 0.05 * cap
            metrics[f"{src}_h{h}"] = {
                "source": src, "horizon_h": h, "capacity_mw": cap,
                "model": reg_metrics(yt, pred, floor), "baseline_persistence": reg_metrics(yt, pers, floor),
                "baseline_same_hour_last_available_day": reg_metrics(yt, lagd, floor),
                "mape_note": "MAPE computed only where actual output > 5% of installed capacity",
                "objective": obj,
            }
            mm = metrics[f"{src}_h{h}"]
            best_base = min(mm["baseline_persistence"]["mae"], mm["baseline_same_hour_last_available_day"]["mae"])
            mm["beats_best_baseline_mae"] = bool(mm["model"]["mae"] < best_base)
            mm["beats_best_baseline_rmse"] = bool(mm["model"]["rmse"] < min(
                mm["baseline_persistence"]["rmse"], mm["baseline_same_hour_last_available_day"]["rmse"]))
            res = pred - yt
            qerr[(src, h)] = (float(np.quantile(res, 0.10)), float(np.quantile(res, 0.90)))
            T_all = np.arange(169, n_hist)
            mf = lgb.LGBMRegressor(**kw).fit(_gen_features(world, src, T_all, h, seed), y[T_all] / cap)
            models[(src, h)] = mf
    return models, metrics, qerr, names


def forecast_generation(world: World, bundle: ForecastBundle, n_hours: int = HORIZON_MAX, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Returns {'mean','lo','hi'} DataFrames (time x source) of MW."""
    n_hist = world.n_hist
    leads = np.arange(1, n_hours + 1)
    T = n_hist - 1 + leads
    mean, lo, hi = {}, {}, {}
    for src in SOURCES_MODELLED:
        _, cap = _source_series(world, src)
        p = np.zeros(n_hours)
        lo_a, hi_a = np.zeros(n_hours), np.zeros(n_hours)
        for h in HORIZONS:
            sel = np.array([lead_to_horizon(int(l)) == h for l in leads])
            if not sel.any():
                continue
            f = _gen_features(world, src, T[sel], h, seed)
            pred = np.clip(bundle.gen_models[(src, h)].predict(f[bundle.feature_names_gen]), 0, 1.05) * cap
            q10, q90 = bundle.gen_err_q[(src, h)]
            p[sel] = pred
            lo_a[sel] = np.clip(pred - max(q90, 0.0), 0, cap)
            hi_a[sel] = np.clip(pred - min(q10, 0.0), 0, cap)
        mean[src], lo[src], hi[src] = p, lo_a, hi_a
    idx = world.times[T]
    imp = world.plant_gen[world.plants.plant_id[world.plants.source == "import"]].sum(axis=1).values
    # scheduled import is a known quantity (assumption: scheduled import profile is known 72 h ahead)
    mean["import"] = lo["import"] = hi["import"] = imp[T]
    return {k: pd.DataFrame(v, index=idx) for k, v in (("mean", mean), ("lo", lo), ("hi", hi))}


def train_forecasts(world: World, seed: int = 42) -> ForecastBundle:
    d_models, scale, d_metrics, d_q, d_names = train_demand(world, seed)
    g_models, g_metrics, g_q, g_names = train_generation(world, seed)
    b = ForecastBundle(version=MODEL_VERSION, demand_models=d_models, gen_models=g_models, demand_scale=scale,
                       demand_rel_err_q=d_q, gen_err_q=g_q, feature_names_demand=d_names, feature_names_gen=g_names)
    b.metrics = {"demand": d_metrics, "generation": g_metrics}
    return b


# ------------------------------------------------------------------------------------------
# predictive maintenance
# ------------------------------------------------------------------------------------------
MAINT_FEATURES = ["vib_mean", "vib_max", "temp_c", "pressure_bar", "flow_rate", "op_hours", "days_since_maint",
                  "faults_last_180d", "load_factor", "vib_slope_7d", "temp_delta_7d", "vib_vs_fleet_z",
                  "temp_vs_fleet_z", "vib_roll7", "temp_roll7", "vibmax_roll7", "pres_roll7", "flow_roll7",
                  "src_geothermal", "src_hydro", "src_wind", "src_thermal"]


def _maint_frame(world: World) -> pd.DataFrame:
    md = world.maint_daily.merge(world.units[["unit_id", "source", "plant_id"]], on="unit_id").sort_values(["unit_id", "day"]).copy()
    g = md.groupby("unit_id")
    md["vib_slope_7d"] = (md.vib_mean - g.vib_mean.shift(7)) / 7
    md["temp_delta_7d"] = md.temp_c - g.temp_c.shift(7)
    for col, nm in (("vib_mean", "vib_roll7"), ("temp_c", "temp_roll7"), ("vib_max", "vibmax_roll7"),
                    ("pressure_bar", "pres_roll7"), ("flow_rate", "flow_roll7")):
        md[nm] = g[col].transform(lambda s_: s_.rolling(7, min_periods=1).mean())
    # fleet-relative z by source and day (uses only same-day cross-section: no future info)
    for col, nm in (("vib_mean", "vib_vs_fleet_z"), ("temp_c", "temp_vs_fleet_z")):
        grp = md.groupby(["source", "day"])[col]
        md[nm] = (md[col] - grp.transform("mean")) / (grp.transform("std") + 1e-6)
    for s in ("geothermal", "hydro", "wind", "thermal"):
        md[f"src_{s}"] = (md.source == s).astype(int)
    md[["vib_slope_7d", "temp_delta_7d", "vib_vs_fleet_z", "temp_vs_fleet_z"]] = md[
        ["vib_slope_7d", "temp_delta_7d", "vib_vs_fleet_z", "temp_vs_fleet_z"]].fillna(0)
    return md


@dataclass
class MaintenanceBundle:
    model: Any
    threshold: float
    metrics: dict
    importances: dict[str, float]
    frame: pd.DataFrame
    version: str = MODEL_VERSION


def train_maintenance(world: World, seed: int = 42) -> MaintenanceBundle:
    md = _maint_frame(world)
    last_labeled = int(md.day.max()) - 14
    test_start = int(last_labeled * 0.78)
    train = md[(md.day < test_start - 14) & (md.day >= 7)]       # 14-day gap avoids label overlap with test
    test = md[(md.day >= test_start) & (md.day <= last_labeled)]
    val_start = int((test_start - 14) * 0.85)
    fit = train[train.day < val_start]
    val = train[train.day >= val_start]
    pos = max(1, int(fit.fault_next_14d.sum()))
    spw = (len(fit) - pos) / pos
    params = dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                  scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=2)
    m = XGBClassifier(**params).fit(fit[MAINT_FEATURES], fit.fault_next_14d)
    pv = m.predict_proba(val[MAINT_FEATURES])[:, 1]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        f = f1_score(val.fault_next_14d, (pv >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    m_eval = XGBClassifier(**params).fit(train[MAINT_FEATURES], train.fault_next_14d)
    pt = m_eval.predict_proba(test[MAINT_FEATURES])[:, 1]
    pred = (pt >= best_t).astype(int)
    fleet_thr = train.vib_mean.quantile(0.9)
    base_pred = (test.vib_mean >= fleet_thr).astype(int)
    prevalence = float(test.fault_next_14d.mean())
    metrics = {
        "label": "fault within next 14 days", "threshold_selected_on_validation": best_t,
        "test_window_days": [test_start, last_labeled], "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_prevalence": prevalence,
        "model": cls_metrics(test.fault_next_14d.values, pred, pt),
        "baseline_vibration_threshold_p90": cls_metrics(test.fault_next_14d.values, base_pred.values,
                                                        test.vib_mean.values),
        "baseline_always_negative": cls_metrics(test.fault_next_14d.values, np.zeros(len(test), int)),
    }
    train_all = md[(md.day >= 7) & (md.day <= last_labeled)]
    final = XGBClassifier(**params).fit(train_all[MAINT_FEATURES], train_all.fault_next_14d)
    imp = dict(sorted(zip(MAINT_FEATURES, map(float, final.feature_importances_)), key=lambda kv: -kv[1]))
    return MaintenanceBundle(model=final, threshold=best_t, metrics=metrics, importances=imp, frame=md)


def maintenance_risk(world: World, mb: MaintenanceBundle) -> pd.DataFrame:
    md = mb.frame
    last = md[md.day == md.day.max()].copy()
    last["risk_probability"] = mb.model.predict_proba(last[MAINT_FEATURES])[:, 1]
    last["flagged"] = (last.risk_probability >= mb.threshold).astype(int)
    fleet = md[md.day == md.day.max()].groupby("source")[["vib_mean", "temp_c", "pressure_bar", "flow_rate"]].agg(["mean", "std"])
    drivers = []
    for r in last.itertuples():
        d = []
        for col, label in (("vib_mean", "vibration"), ("temp_c", "temperature"), ("pressure_bar", "pressure"), ("flow_rate", "flow")):
            mu, sd = fleet.loc[r.source, (col, "mean")], fleet.loc[r.source, (col, "std")]
            z = (getattr(r, col) - mu) / (sd + 1e-6)
            if abs(z) >= 1.3:
                d.append(f"{label} {z:+.1f}σ vs {r.source} fleet")
        if r.days_since_maint > 100:
            d.append(f"{int(r.days_since_maint)} days since last maintenance")
        if r.faults_last_180d >= 2:
            d.append(f"{int(r.faults_last_180d)} faults in last 180 days")
        drivers.append(d)
    last["drivers"] = drivers
    return last[["unit_id", "plant_id", "source", "risk_probability", "flagged", "vib_mean", "temp_c", "pressure_bar",
                 "flow_rate", "op_hours", "days_since_maint", "faults_last_180d", "drivers"]].sort_values(
        "risk_probability", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------------
# anomaly detection
# ------------------------------------------------------------------------------------------
CHANNELS = ["vibration", "temperature", "pressure", "flow"]


@dataclass
class AnomalyBundle:
    model: IsolationForest
    stats: dict[str, dict]
    threshold: float
    z_threshold: float
    metrics: dict
    version: str = MODEL_VERSION


def _anomaly_features(df: pd.DataFrame, stats: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    feats, zraw = [], []
    for aid, g in df.groupby("asset_id", sort=False):
        s = stats[aid]
        cols = []
        zs = []
        for ch in CHANNELS:
            a, b = s["fit"][ch]
            resid = (g[ch].values - (a + b * g.load_pct.values))
            cols.append(resid / s["resid_sd"][ch])
            zs.append((g[ch].values - s["mean"][ch]) / s["sd"][ch])
        feats.append(np.column_stack(cols + [(g.load_pct.values - s["load_mean"]) / s["load_sd"]]))
        zraw.append(np.column_stack(zs))
    return np.vstack(feats), np.vstack(zraw)


def train_anomaly(world: World, seed: int = 42) -> AnomalyBundle:
    tel = world.telemetry.sort_values(["asset_id", "time"]).reset_index(drop=True)
    times = np.sort(tel.time.unique())
    cut = times[int(len(times) * 0.6)]
    tr, te = tel[tel.time < cut], tel[tel.time >= cut]
    stats = {}
    for aid, g in tr.groupby("asset_id"):
        fit, rsd, mean, sd = {}, {}, {}, {}
        for ch in CHANNELS:
            b, a = np.polyfit(g.load_pct.values, g[ch].values, 1)
            fit[ch] = (float(a), float(b))
            rsd[ch] = float(np.std(g[ch].values - (a + b * g.load_pct.values)) + 1e-9)
            mean[ch], sd[ch] = float(g[ch].mean()), float(g[ch].std() + 1e-9)
        stats[aid] = {"fit": fit, "resid_sd": rsd, "mean": mean, "sd": sd,
                      "load_mean": float(g.load_pct.mean()), "load_sd": float(g.load_pct.std() + 1e-9)}
    Xtr, ztr = _anomaly_features(tr, stats)
    Xte, zte = _anomaly_features(te, stats)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=2).fit(Xtr)
    s_tr = -iso.score_samples(Xtr)
    thr = float(np.quantile(s_tr, 0.95))
    s_te = -iso.score_samples(Xte)
    # evaluation ordering must follow groupby(sort=False) order == tel order within asset
    order = pd.concat([g for _, g in te.groupby("asset_id", sort=False)])
    y = order.is_anomaly.values
    pred = (s_te >= thr).astype(int)
    z_thr = 3.5
    zpred = (np.max(np.abs(zte), axis=1) >= z_thr).astype(int)
    metrics = {"train_window_end": str(cut), "test_rows": int(len(te)), "test_anomaly_rate": float(y.mean()),
               "isolation_forest": cls_metrics(y, pred, s_te),
               "baseline_zscore_3.5": cls_metrics(y, zpred, np.max(np.abs(zte), axis=1)),
               "threshold_rule": "95th percentile of training anomaly scores"}
    return AnomalyBundle(model=iso, stats=stats, threshold=thr, z_threshold=z_thr, metrics=metrics)


def recent_anomalies(world: World, ab: AnomalyBundle, last_hours: int = 72) -> pd.DataFrame:
    tel = world.telemetry.sort_values(["asset_id", "time"]).reset_index(drop=True)
    X, z = _anomaly_features(tel, ab.stats)
    ordered = pd.concat([g for _, g in tel.groupby("asset_id", sort=False)]).reset_index(drop=True)
    ordered["score"] = -ab.model.score_samples(X)
    ordered["flag"] = (ordered.score >= ab.threshold).astype(int)
    drv = np.argmax(np.abs(X[:, :4]), axis=1)
    ordered["driver_channel"] = [CHANNELS[i] for i in drv]
    t_end = ordered.time.max()
    rec = ordered[(ordered.time > t_end - pd.Timedelta(hours=last_hours)) & (ordered.flag == 1)]
    return rec[["time", "asset_id", "asset_type", "score", "driver_channel", "is_anomaly"]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------
# electricity loss detection
# ------------------------------------------------------------------------------------------
@dataclass
class LossBundle:
    model: IsolationForest
    threshold: float
    metrics: dict
    latest: pd.DataFrame
    version: str = MODEL_VERSION


def _loss_frame(world: World) -> pd.DataFrame:
    fm = world.feeder_monthly.sort_values(["feeder_id", "month"]).copy()
    fm["unaccounted_pct"] = 1 - fm.metered_mwh / fm.input_mwh
    fm["ntl_est_pct"] = fm.unaccounted_pct - fm.tech_loss_est_pct
    g = fm.groupby("feeder_id")
    fm["ntl_roll3"] = g.ntl_est_pct.transform(lambda s: s.rolling(3, min_periods=1).mean())
    grp = fm.groupby(["region", "month"]).ntl_est_pct
    med = grp.transform("median")
    mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s)))) * 1.4826 + 1e-4
    fm["peer_z"] = (fm.ntl_est_pct - med) / mad
    fm["metered_per_cust"] = fm.metered_mwh * 1000 / fm.n_customers
    fm["own_hist_z"] = (fm.metered_per_cust - g.metered_per_cust.transform(lambda s: s.expanding().mean().shift(1))) / (
        g.metered_per_cust.transform(lambda s: s.expanding().std().shift(1)) + 1e-6)
    fm[["own_hist_z"]] = fm[["own_hist_z"]].fillna(0)
    return fm


LOSS_FEATURES = ["ntl_est_pct", "ntl_roll3", "peer_z", "load_ratio", "own_hist_z"]


def train_loss(world: World, seed: int = 42) -> LossBundle:
    fm = _loss_frame(world)
    tr, te = fm[fm.month <= 8], fm[fm.month >= 9]
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=2).fit(tr[LOSS_FEATURES])
    thr = float(np.quantile(-iso.score_samples(tr[LOSS_FEATURES]), 0.92))
    s = -iso.score_samples(te[LOSS_FEATURES])
    pred = (s >= thr).astype(int)
    base = (te.ntl_est_pct >= 0.05).astype(int)
    hyb = ((s >= thr) & (te.ntl_est_pct.values >= 0.04)).astype(int)
    metrics = {"train_months": [0, 8], "test_months": [9, 11], "test_positive_rate": float(te.ntl_active.mean()),
               "isolation_forest": cls_metrics(te.ntl_active.values, pred, s),
               "baseline_ntl_est_ge_5pct": cls_metrics(te.ntl_active.values, base.values, te.ntl_est_pct.values),
               "hybrid_iforest_and_ntl_ge_4pct": cls_metrics(te.ntl_active.values, hyb, s),
               "label": "feeder has active injected non-technical loss in the month"}
    last = fm[fm.month == fm.month.max()].copy()
    last["anomaly_score"] = -iso.score_samples(last[LOSS_FEATURES])
    last["flagged"] = ((last.anomaly_score >= thr) & (last.ntl_est_pct >= 0.04)).astype(int)
    last["ntl_mwh_est"] = (last.ntl_est_pct.clip(lower=0) * last.input_mwh)
    last["tech_mwh_est"] = (last.tech_loss_est_pct.clip(lower=0) * last.input_mwh)
    return LossBundle(model=iso, threshold=thr, metrics=metrics, latest=last.reset_index(drop=True))
