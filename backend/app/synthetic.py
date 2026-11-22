"""Reproducible synthetic data generator for POWERIQ Kenya.

Everything produced here is SYNTHETIC. Plant names and rough locations reference real Kenyan
facilities so the digital twin is recognisable, but capacities, telemetry, demand, faults and
losses are generated, not measured. Every table carries `data_origin = "synthetic"` and the
generation method is stored in `World.provenance`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import geo

SEED = 42
N_HIST_DAYS = 270
HORIZON_MAX = 72
# Last observed hour is 2026-11-18 23:00 EAT; the world continues 72 h "into the future".
HIST_END = pd.Timestamp("2026-11-19 00:00")  # exclusive
N_HIST = N_HIST_DAYS * 24
N_TOTAL = N_HIST + HORIZON_MAX
START = HIST_END - pd.Timedelta(hours=N_HIST)

HOLIDAYS = {
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01", "2026-06-01",
    "2026-10-20", "2026-12-12", "2026-12-25", "2026-12-26",
}

SOURCES = ["geothermal", "hydro", "wind", "solar", "thermal", "import"]

# id, name, source, capacity MW (approximate, illustrative), lon, lat, n_units
PLANT_DEFS: list[tuple] = [
    ("OLK1", "Olkaria I", "geothermal", 185, 36.290, -0.900, 6),
    ("OLK2", "Olkaria II", "geothermal", 105, 36.300, -0.890, 3),
    ("OLK4", "Olkaria IV", "geothermal", 150, 36.310, -0.910, 2),
    ("OLK5", "Olkaria V", "geothermal", 158, 36.280, -0.920, 2),
    ("OLK1AU", "Olkaria I AU", "geothermal", 150, 36.295, -0.905, 2),
    ("MENG", "Menengai", "geothermal", 105, 36.070, -0.200, 3),
    ("TURK", "Turkwel", "hydro", 106, 35.360, 1.970, 2),
    ("GITA", "Gitaru", "hydro", 225, 37.620, -0.780, 2),
    ("KAMB", "Kamburu", "hydro", 94, 37.680, -0.790, 2),
    ("MASI", "Masinga", "hydro", 40, 37.570, -0.880, 2),
    ("KIND", "Kindaruma", "hydro", 72, 37.770, -0.800, 3),
    ("KIAM", "Kiambere", "hydro", 168, 37.950, -0.700, 2),
    ("SOND", "Sondu Miriu", "hydro", 60, 34.800, -0.370, 2),
    ("SANG", "Sang'oro", "hydro", 21, 34.850, -0.350, 2),
    ("WANJ", "Wanjii", "hydro", 20, 37.550, -0.860, 2),
    ("LTWP", "Lake Turkana Wind", "wind", 310, 36.720, 2.500, 12),
    ("KIPE", "Kipeto Wind", "wind", 100, 36.650, -1.620, 4),
    ("NGON", "Ngong Hills Wind", "wind", 25, 36.640, -1.460, 3),
    ("GARS", "Garissa Solar", "solar", 55, 39.650, -0.420, 1),
    ("MALI", "Malindi Solar", "solar", 40, 40.100, -3.000, 1),
    ("KIP3", "Kipevu III", "thermal", 120, 39.640, -4.050, 3),
    ("KIP1", "Kipevu I", "thermal", 75, 39.650, -4.045, 2),
    ("RABA", "Rabai", "thermal", 90, 39.600, -4.000, 4),
    ("THIK", "Thika Power", "thermal", 87, 37.080, -1.040, 3),
    ("EMBG", "Embakasi GT", "thermal", 60, 36.900, -1.320, 2),
    ("MUHO", "Muhoroni GT", "thermal", 30, 35.200, -0.150, 2),
    ("ETHI", "Ethiopia HVDC Import (Suswa)", "import", 200, 36.350, -1.080, 1),
]

# id, name, lon, lat, kV
SUBSTATION_DEFS: list[tuple] = [
    ("EMBAKASI", "Embakasi", 36.90, -1.32, 220), ("NBI_N", "Nairobi North", 36.86, -1.22, 132),
    ("SUSWA", "Suswa", 36.35, -1.08, 400), ("OLKARIA", "Olkaria", 36.29, -0.90, 220),
    ("NAKURU", "Nakuru / Menengai", 36.07, -0.28, 132), ("LESSOS", "Lessos", 35.30, 0.05, 220),
    ("ELDORET", "Eldoret", 35.27, 0.52, 132), ("KISUMU", "Kisumu", 34.76, -0.10, 132),
    ("KISII", "Kisii", 34.77, -0.68, 132), ("THIKA", "Thika", 37.08, -1.04, 220),
    ("KAMBURU", "Kamburu (Tana)", 37.68, -0.79, 220), ("MERU", "Meru", 37.65, 0.05, 132),
    ("MACHAKOS", "Machakos", 37.26, -1.52, 132), ("RABAI", "Rabai (Mombasa)", 39.57, -3.98, 220),
    ("GARISSA", "Garissa", 39.65, -0.45, 132), ("LTWP_HUB", "Loiyangalani (LTWP)", 36.72, 2.50, 220),
    ("TURKWEL", "Turkwel", 35.36, 1.97, 132), ("MALINDI", "Malindi", 40.10, -3.00, 66),
]

# from, to, capacity MW (illustrative), kV
LINE_DEFS: list[tuple] = [
    ("OLKARIA", "SUSWA", 900, 220), ("SUSWA", "EMBAKASI", 1000, 400), ("OLKARIA", "NAKURU", 650, 220),
    ("NAKURU", "LESSOS", 450, 220), ("LESSOS", "ELDORET", 250, 132), ("LESSOS", "KISUMU", 400, 220),
    ("KISUMU", "KISII", 200, 132), ("EMBAKASI", "NBI_N", 700, 132), ("EMBAKASI", "THIKA", 450, 220),
    ("THIKA", "KAMBURU", 500, 220), ("KAMBURU", "MERU", 320, 132), ("EMBAKASI", "MACHAKOS", 220, 132),
    ("MACHAKOS", "KAMBURU", 260, 132), ("EMBAKASI", "RABAI", 700, 400), ("RABAI", "MALINDI", 90, 66),
    ("KAMBURU", "GARISSA", 110, 132), ("LTWP_HUB", "SUSWA", 330, 400), ("LESSOS", "TURKWEL", 130, 132),
    ("SUSWA", "NAKURU", 500, 220), ("KISII", "NAKURU", 180, 132),
]


@dataclass
class World:
    seed: int
    times: pd.DatetimeIndex
    n_hist: int
    regions: pd.DataFrame
    cells: pd.DataFrame
    plants: pd.DataFrame
    substations: pd.DataFrame
    lines: pd.DataFrame
    weather: pd.DataFrame          # long: time, region, temp_c, ghi_wm2, wind_ms, cloud, rain_mm
    demand: pd.DataFrame           # wide: time x region (MW, regional input demand)
    plant_gen: pd.DataFrame        # wide: time x plant_id (MW, after curtailment)
    plant_potential: pd.DataFrame  # wide: potential MW before curtailment / dispatch
    plant_avail: pd.DataFrame      # wide: fraction of capacity available (planned/forced outages)
    system: pd.DataFrame           # time-indexed system quantities
    curtailment_events: pd.DataFrame
    units: pd.DataFrame
    maint_daily: pd.DataFrame
    faults: pd.DataFrame
    telemetry: pd.DataFrame
    feeders: pd.DataFrame
    feeder_monthly: pd.DataFrame
    grid_incidents: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0, sigma * math.sqrt(1 - phi ** 2), n)
    return lfilter([1.0], [1.0, -phi], e)


def _gauss(h, mu, s):
    d = np.minimum(np.abs(h - mu), 24 - np.abs(h - mu))
    return np.exp(-0.5 * (d / s) ** 2)


def _daily_shape(hour: np.ndarray, resid: float, comm: float) -> np.ndarray:
    s = (0.62 + 0.16 * _gauss(hour, 7.5, 1.6) + 0.12 * comm * _gauss(hour, 12.5, 2.8)
         + 0.40 * resid * _gauss(hour, 19.5, 2.0) - 0.14 * _gauss(hour, 3.5, 2.5))
    return s


def _make_weather(times: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(times)
    hour = times.hour.values + times.minute.values / 60
    doy = times.dayofyear.values
    rows = []
    # shared synoptic wind / cloud drivers so regions are correlated
    synoptic_cloud = _ar1(n, 0.985, 0.30, rng)
    synoptic_wind = _ar1(n, 0.97, 0.9, rng)
    for rg in geo.REGIONS:
        seasonal_t = 2.0 * np.sin(2 * np.pi * (doy - 60) / 365)
        diurnal_t = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24)
        temp = rg.temp_base_c + seasonal_t + diurnal_t + _ar1(n, 0.98, 1.2, rng)
        rainy = 0.5 * (np.exp(-0.5 * ((doy - 110) / 28) ** 2) + np.exp(-0.5 * ((doy - 310) / 25) ** 2))
        cloud = np.clip(rg.cloud_base + 0.35 * rainy + 0.45 * synoptic_cloud + _ar1(n, 0.97, 0.12, rng)
                        + 0.08 * _gauss(hour, 15.5, 3.0), 0.0, 1.0)
        clear = np.clip(np.sin(np.pi * (hour - 6.3) / 12.0), 0, None) ** 1.1
        ghi = 1000.0 * clear * (1 - 0.75 * cloud)
        seasonal_w = 1.0 + 0.30 * np.exp(-0.5 * ((doy - 210) / 45) ** 2)
        wind = rg.wind_base_ms * seasonal_w * (1 + 0.28 * _gauss(hour, 3.0, 4.0) - 0.12 * _gauss(hour, 13.0, 3.0))
        wind = np.clip(wind + 0.55 * synoptic_wind + _ar1(n, 0.95, 0.9, rng), 0.2, 26.0)
        rain = np.clip((cloud - 0.55) * 4.0 * rainy * rng.gamma(1.2, 1.0, n), 0, None)
        rows.append(pd.DataFrame({"time": times, "region": rg.id, "temp_c": temp, "ghi_wm2": ghi,
                                  "wind_ms": wind, "cloud": cloud, "rain_mm": rain}))
    return pd.concat(rows, ignore_index=True)


def _wind_power_curve(v: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    m = (v >= 3.0) & (v < 12.0)
    out[m] = ((v[m] - 3.0) / 9.0) ** 3
    out[(v >= 12.0) & (v < 25.0)] = 1.0
    return out


def make_world(seed: int = SEED) -> World:
    rng = np.random.default_rng(seed)
    times = pd.date_range(START, periods=N_TOTAL, freq="h")
    hour = times.hour.values.astype(float)
    dow = times.dayofweek.values
    date_str = times.strftime("%Y-%m-%d")
    is_hol = np.isin(date_str, list(HOLIDAYS))

    # ---- structure ------------------------------------------------------------------------
    cells_list = geo.build_cells()
    cells = pd.DataFrame([{k: v for k, v in c.items()} for c in cells_list])
    cells["data_origin"] = "synthetic"
    regions = pd.DataFrame([{**r.__dict__, "data_origin": "synthetic"} for r in geo.REGIONS])

    cell_pts = list(zip(cells.lon, cells.lat))
    plants = pd.DataFrame(PLANT_DEFS, columns=["plant_id", "name", "source", "capacity_mw", "lon", "lat", "n_units"])
    plants["region"] = [geo.REGIONS[geo.nearest_index(a, b, [(r.lon, r.lat) for r in geo.REGIONS])].id
                        for a, b in zip(plants.lon, plants.lat)]
    plants["cell_id"] = [cells.cell_id.iloc[geo.nearest_index(a, b, cell_pts)] for a, b in zip(plants.lon, plants.lat)]
    sub = pd.DataFrame(SUBSTATION_DEFS, columns=["bus_id", "name", "lon", "lat", "kv"])
    sub_pts = list(zip(sub.lon, sub.lat))
    plants["bus_id"] = [sub.bus_id.iloc[geo.nearest_index(a, b, sub_pts)] for a, b in zip(plants.lon, plants.lat)]
    cells["bus_id"] = [sub.bus_id.iloc[geo.nearest_index(a, b, sub_pts)] for a, b in cell_pts]
    plants["data_origin"] = "synthetic"
    sub["data_origin"] = "synthetic"
    bus_xy = {r.bus_id: (r.lon, r.lat) for r in sub.itertuples()}
    lines = pd.DataFrame(LINE_DEFS, columns=["from_bus", "to_bus", "capacity_mw", "kv"])
    lines["line_id"] = [f"{a}-{b}" for a, b in zip(lines.from_bus, lines.to_bus)]
    lines["length_km"] = [geo.haversine_km(*bus_xy[a], *bus_xy[b]) * 1.15 for a, b in zip(lines.from_bus, lines.to_bus)]
    lines["x_pu"] = lines.length_km / 100.0
    lines["data_origin"] = "synthetic"

    # ---- weather --------------------------------------------------------------------------
    weather = _make_weather(times, rng)
    wide = {v: weather.pivot(index="time", columns="region", values=v).reindex(times)[geo.REGION_IDS]
            for v in ["temp_c", "ghi_wm2", "wind_ms", "cloud", "rain_mm"]}

    # ---- regional demand ------------------------------------------------------------------
    demand = {}
    for rg in geo.REGIONS:
        shape = _daily_shape(hour, rg.residential, rg.commercial)
        shape = shape / _daily_shape(np.arange(24.0), rg.residential, rg.commercial).max()
        weekend = np.where(dow == 6, 0.88, np.where(dow == 5, 0.94, 1.0))
        weekend = 1 - (1 - weekend) * rg.commercial * 1.2 - (1 - weekend) * 0.0
        hol = np.where(is_hol, 0.86, 1.0)
        trend = 1.0 + 0.00001 * np.arange(N_TOTAL)  # slow growth (~7% over the window)
        temp_eff = 1 + 0.006 * (wide["temp_c"][rg.id].values - rg.temp_base_c)
        noise = np.exp(_ar1(N_TOTAL, 0.92, 0.03, rng) + rng.normal(0, 0.008, N_TOTAL))
        demand[rg.id] = rg.base_peak_mw * 0.84 * shape * weekend * hol * trend * temp_eff * noise
    demand = pd.DataFrame(demand, index=times)
    total_demand = demand.sum(axis=1).values

    # ---- generation potential and dispatch ------------------------------------------------
    doy = times.dayofyear.values
    rain_nat = wide["rain_mm"].mean(axis=1).values
    # reservoir water factor: slow response to 45-day smoothed rainfall, 0.5 .. 0.95
    kern = np.ones(24 * 45) / (24 * 45)
    water = np.convolve(np.pad(rain_nat, (len(kern) - 1, 0), mode="edge"), kern, mode="valid")
    wf = np.clip(0.50 + 0.9 * (water / (water.max() + 1e-9)), 0.50, 0.95) + 0.03 * _ar1(N_TOTAL, 0.995, 1.0, rng)
    wf = np.clip(wf, 0.45, 0.95)

    pot = {}
    outage_frac = {}
    for p in plants.itertuples():
        # forced/planned outage as fraction of capacity unavailable (piecewise constant episodes)
        avail = np.ones(N_TOTAL)
        n_ep = int(rng.integers(1, 4))
        for _ in range(n_ep):
            s0 = int(rng.integers(0, N_TOTAL - 200))
            dur = int(rng.integers(24, 240))
            avail[s0:s0 + dur] = 1 - (1.0 / p.n_units) * rng.integers(1, max(2, min(3, p.n_units)) + 1)
        avail = np.clip(avail, 0, 1)
        outage_frac[p.plant_id] = avail
        r = p.region
        if p.source == "geothermal":
            g = p.capacity_mw * 0.94 * avail * (1 + rng.normal(0, 0.006, N_TOTAL))
        elif p.source == "hydro":
            g = p.capacity_mw * avail * wf * (0.80 + 0.2 * rng.random())
        elif p.source == "wind":
            v = wide["wind_ms"][r].values * (1.12 if p.plant_id == "LTWP" else 1.0)
            v = np.clip(v + rng.normal(0, 0.4, N_TOTAL), 0, 30)
            g = p.capacity_mw * _wind_power_curve(v) * avail * 0.97
        elif p.source == "solar":
            g = p.capacity_mw * (wide["ghi_wm2"][r].values / 1000.0) * 0.9 * avail
        elif p.source == "import":
            g = p.capacity_mw * 0.78 * avail * (1 + rng.normal(0, 0.02, N_TOTAL))
        else:  # thermal handled by dispatch; potential = capacity * availability
            g = p.capacity_mw * avail
        pot[p.plant_id] = np.clip(g, 0, p.capacity_mw)
    pot = pd.DataFrame(pot, index=times)
    avail_df = pd.DataFrame(outage_frac, index=times)

    def sum_src(df, src):
        ids = plants.plant_id[plants.source == src].tolist()
        return df[ids].sum(axis=1).values

    tloss = 0.025
    required = total_demand * (1 + tloss)
    geo_pot, imp_pot = sum_src(pot, "geothermal"), sum_src(pot, "import")
    hydro_avail = sum_src(pot, "hydro")
    wind_pot, solar_pot = sum_src(pot, "wind"), sum_src(pot, "solar")
    thermal_cap = sum_src(pot, "thermal")

    # transmission export limit on the Lake Turkana corridor (curtailment driver)
    corridor_limit = 250.0
    ltwp_pot = pot["LTWP"].values
    corridor_curt = np.clip(ltwp_pot - corridor_limit, 0, None)
    wind_after_corridor = wind_pot - corridor_curt
    # surplus backs off imports first, then geothermal steam is vented; hydro minimum is soft
    imp_used = np.minimum(imp_pot, np.clip(required - geo_pot, 0, None))
    geo_used = np.minimum(geo_pot, required)
    non_flex = geo_used + imp_used
    hydro_min = np.minimum(0.30 * hydro_avail, np.clip(required - non_flex, 0, None))
    vre = wind_after_corridor + solar_pot
    excess = np.clip(non_flex + hydro_min + vre - required, 0, None)
    sys_curt = np.minimum(excess, vre)
    vre_scale = np.where(vre > 0, 1 - sys_curt / np.maximum(vre, 1e-9), 1.0)
    wind_actual = wind_after_corridor * vre_scale
    solar_actual = solar_pot * vre_scale
    resid = required - non_flex - wind_actual - solar_actual
    hydro = np.clip(resid, hydro_min, hydro_avail)
    thermal = np.clip(required - non_flex - wind_actual - solar_actual - hydro, 0, thermal_cap)
    supplied = non_flex + wind_actual + solar_actual + hydro + thermal
    unserved = np.clip(required - supplied, 0, None)

    gen = pd.DataFrame(index=times, columns=plants.plant_id, dtype=float)
    for p in plants.itertuples():
        pid = p.plant_id
        if p.source == "geothermal":
            gen[pid] = pot[pid].values * np.where(geo_pot > 0, geo_used / np.maximum(geo_pot, 1e-9), 1.0)
        elif p.source == "import":
            gen[pid] = pot[pid].values * np.where(imp_pot > 0, imp_used / np.maximum(imp_pot, 1e-9), 1.0)
        elif p.source == "hydro":
            share = pot[pid].values / np.maximum(hydro_avail, 1e-9)
            gen[pid] = hydro * share
        elif p.source == "thermal":
            share = pot[pid].values / np.maximum(thermal_cap, 1e-9)
            gen[pid] = thermal * share
        elif p.source == "wind":
            if pid == "LTWP":
                gen[pid] = (pot[pid].values - corridor_curt) * vre_scale
            else:
                gen[pid] = pot[pid].values * vre_scale
        elif p.source == "solar":
            gen[pid] = pot[pid].values * vre_scale
    gen = gen.astype(float)

    # curtailment events (hourly rows merged into episodes later at query time)
    curt_rows = []
    for p in plants[plants.source.isin(["wind", "solar"])].itertuples():
        c = np.clip(pot[p.plant_id].values - gen[p.plant_id].values, 0, None)
        idx = np.where(c > 1.0)[0]
        for i in idx:
            reason = "transmission_corridor_limit" if (p.plant_id == "LTWP" and corridor_curt[i] > 1.0) else "system_oversupply"
            curt_rows.append((times[i], p.plant_id, p.source, round(float(c[i]), 2), reason))
    curtail = pd.DataFrame(curt_rows, columns=["time", "plant_id", "source", "curtailed_mw", "reason"])
    curtail["data_origin"] = "synthetic"

    system = pd.DataFrame({
        "total_demand_mw": total_demand, "required_gen_mw": required, "supplied_mw": supplied,
        "unserved_mw": unserved, "geothermal_mw": sum_src(gen, "geothermal"), "hydro_mw": sum_src(gen, "hydro"),
        "wind_mw": sum_src(gen, "wind"), "solar_mw": sum_src(gen, "solar"), "thermal_mw": sum_src(gen, "thermal"),
        "import_mw": sum_src(gen, "import"), "water_factor": wf, "thermal_cap_mw": thermal_cap,
        "hydro_avail_mw": hydro_avail,
    }, index=times)

    # ---- unit health, faults, daily snapshots ---------------------------------------------
    units, maint_daily, faults = _make_unit_history(plants, rng)

    # ---- hourly telemetry for anomaly detection -------------------------------------------
    telemetry = _make_telemetry(rng)

    # ---- feeders and monthly energy accounting --------------------------------------------
    feeders, feeder_monthly = _make_feeders(cells, rng)

    grid_incidents = _make_incidents(times, sub, lines, rng)

    prov = {
        "data_origin": "synthetic",
        "generator": "poweriq.synthetic.make_world",
        "seed": seed,
        "history_hours": N_HIST,
        "future_hours_generated": HORIZON_MAX,
        "hist_end": str(HIST_END),
        "timezone": "Africa/Nairobi (naive local timestamps, UTC+03:00)",
        "method": ("Stochastic weather (AR(1) + seasonal/diurnal terms), regional demand from daily shape x calendar x "
                   "weather x AR noise, plant output from physical curves (wind power curve, GHI, water factor) with "
                   "random availability outages, merit-order dispatch with curtailment, latent-health degradation "
                   "model for unit faults, injected anomalies in hourly telemetry, injected non-technical losses on feeders."),
        "caveat": "Plant names reference real facilities for recognisability; capacities and all time series are synthetic.",
    }
    return World(seed=seed, times=times, n_hist=N_HIST, regions=regions, cells=cells, plants=plants, plant_avail=avail_df,
                 substations=sub, lines=lines, weather=weather, demand=demand, plant_gen=gen, plant_potential=pot,
                 system=system, curtailment_events=curtail, units=units, maint_daily=maint_daily, faults=faults,
                 telemetry=telemetry, feeders=feeders, feeder_monthly=feeder_monthly, grid_incidents=grid_incidents,
                 provenance=prov)


# -------------------------------------------------------------------------------------------
def _make_unit_history(plants: pd.DataFrame, rng: np.random.Generator):
    unit_rows = []
    for p in plants.itertuples():
        if p.source in ("solar", "import"):
            continue
        for k in range(p.n_units):
            unit_rows.append({"unit_id": f"{p.plant_id}-U{k + 1}", "plant_id": p.plant_id, "source": p.source,
                              "rated_mw": p.capacity_mw / p.n_units})
    units = pd.DataFrame(unit_rows)
    units["data_origin"] = "synthetic"
    base = {"geothermal": dict(vib=2.2, temp=88, pres=12.0, flow=95),
            "hydro": dict(vib=1.8, temp=62, pres=18.0, flow=140),
            "wind": dict(vib=1.4, temp=55, pres=3.0, flow=0.0),
            "thermal": dict(vib=2.8, temp=94, pres=22.0, flow=70)}
    n_days = N_HIST_DAYS
    rows, fault_rows = [], []
    for u in units.itertuples():
        b = base[u.source]
        health = rng.uniform(0.55, 1.0)
        lam = rng.lognormal(math.log(0.004), 0.45)
        since_maint = int(rng.integers(0, 90))
        op_hours = float(rng.uniform(5000, 60000))
        fault_days: list[int] = []
        for d in range(n_days):
            load = rng.uniform(0.55, 0.98)
            health -= lam * (0.7 + 0.6 * load) * rng.lognormal(0, 0.25)
            since_maint += 1
            op_hours += 24 * load
            if since_maint >= 100 and rng.random() < 0.08:  # planned maintenance
                health = min(1.0, health + rng.uniform(0.25, 0.5))
                since_maint = 0
            hazard = 0.0008 + 0.12 * max(0.0, 0.55 - health) ** 1.5 * 6
            fault = (health < 0.12) or (rng.random() < hazard)
            wear = max(0.0, 1 - health)
            vib = b["vib"] * (1 + 1.5 * wear ** 2) + rng.normal(0, 0.12)
            vib_max = vib * (1.25 + 0.8 * wear) + abs(rng.normal(0, 0.15))
            temp = b["temp"] + 18 * wear + 6 * load + rng.normal(0, 1.2)
            pres = b["pres"] * (1 - 0.18 * wear) + rng.normal(0, 0.15)
            flow = b["flow"] * (1 - 0.10 * wear) * (0.6 + 0.4 * load) + rng.normal(0, 0.8)
            rows.append({"unit_id": u.unit_id, "day": d, "date": START.normalize() + pd.Timedelta(days=d),
                         "vib_mean": vib, "vib_max": vib_max, "temp_c": temp, "pressure_bar": pres,
                         "flow_rate": flow, "op_hours": op_hours, "days_since_maint": since_maint,
                         "faults_last_180d": sum(1 for fd in fault_days if d - fd <= 180), "load_factor": load,
                         "latent_health": health})
            if fault:
                fault_days.append(d)
                fault_rows.append({"unit_id": u.unit_id, "day": d, "date": START.normalize() + pd.Timedelta(days=d),
                                   "severity": float(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))})
                health = rng.uniform(0.8, 1.0)  # repair
                since_maint = 0
    md = pd.DataFrame(rows)
    faults = pd.DataFrame(fault_rows)
    faults["data_origin"] = "synthetic"
    # label: fault in the next 14 days (exclusive of today)
    md["fault_next_14d"] = 0
    fset = {(r.unit_id, r.day) for r in faults.itertuples()}
    lab = np.zeros(len(md), dtype=int)
    for i, (uid, d) in enumerate(zip(md.unit_id.values, md.day.values)):
        lab[i] = int(any((uid, d + k) in fset for k in range(1, 15)))
    md["fault_next_14d"] = lab
    md["data_origin"] = "synthetic"
    return units, md, faults


def _make_telemetry(rng: np.random.Generator) -> pd.DataFrame:
    n = 24 * 30
    t = pd.date_range(HIST_END - pd.Timedelta(hours=n), periods=n, freq="h")
    hour = t.hour.values.astype(float)
    frames = []
    for a in range(16):
        atype = "turbine" if a < 8 else "transformer"
        load = np.clip(0.62 + 0.28 * _gauss(hour, 19.5, 3.0) - 0.15 * _gauss(hour, 3.5, 3.0)
                       + _ar1(n, 0.9, 0.05, rng), 0.2, 1.0) * 100
        v0, t0, p0, f0 = (2.0, 70.0, 14.0, 100.0) if atype == "turbine" else (0.8, 55.0, 1.2, 40.0)
        vib = v0 + 0.004 * load + rng.normal(0, 0.05, n)
        temp = t0 + 0.22 * load + rng.normal(0, 0.8, n)
        pres = p0 * (1 + 0.0006 * (load - 60)) + rng.normal(0, 0.05 * p0 / 10, n)
        flow = f0 * (0.5 + 0.005 * load) + rng.normal(0, 0.6, n)
        label = np.zeros(n, dtype=int)
        kind = np.array([""] * n, dtype=object)
        n_ev = int(rng.integers(3, 7))
        for _ in range(n_ev):
            s0 = int(rng.integers(40, n - 30))
            dur = int(rng.integers(4, 14))
            sl = slice(s0, s0 + dur)
            typ = rng.choice(["bearing_wear", "overheating", "pressure_drop", "stuck_sensor"])
            ramp = np.linspace(0.3, 1.0, dur)
            if typ == "bearing_wear":
                vib[sl] += ramp * rng.uniform(1.0, 2.2) * v0
            elif typ == "overheating":
                temp[sl] += ramp * rng.uniform(10, 24)
            elif typ == "pressure_drop":
                pres[sl] *= 1 - ramp * rng.uniform(0.12, 0.28)
            else:
                flow[sl] = flow[s0]
            label[sl] = 1
            kind[sl] = typ
        frames.append(pd.DataFrame({"time": t, "asset_id": f"{'TUR' if a < 8 else 'TRF'}-{a % 8 + 1:02d}",
                                    "asset_type": atype, "load_pct": load, "vibration": vib, "temperature": temp,
                                    "pressure": pres, "flow": flow, "is_anomaly": label, "anomaly_kind": kind}))
    df = pd.concat(frames, ignore_index=True)
    df["data_origin"] = "synthetic"
    return df


def _make_feeders(cells: pd.DataFrame, rng: np.random.Generator):
    rows = []
    for c in cells.itertuples():
        for k in range(3):
            rows.append({"feeder_id": f"{c.cell_id}-F{k + 1}", "cell_id": c.cell_id, "region": c.region,
                         "n_customers": int(rng.integers(300, 4000)),
                         "line_resistance": rng.uniform(0.6, 1.6),
                         "base_load_mw": max(0.15, (c.share * 1.0) * rng.uniform(0.5, 1.5) * 12.0 / 3.0 * 0.3)})
    feeders = pd.DataFrame(rows)
    feeders["theft_start_month"] = np.where(rng.random(len(feeders)) < 0.085, rng.integers(3, 9, len(feeders)), -1)
    feeders["ntl_true_pct"] = np.where(feeders.theft_start_month >= 0, rng.uniform(0.06, 0.26, len(feeders)), 0.0)
    feeders["data_origin"] = "synthetic"
    mrows = []
    for f in feeders.itertuples():
        for m in range(12):
            load = f.base_load_mw * (1 + 0.06 * math.sin(2 * math.pi * m / 12)) * rng.lognormal(0, 0.04)
            input_mwh = load * 730.0
            load_ratio = min(1.2, load / (f.base_load_mw * 1.25))
            tech_pct = 0.025 + 0.06 * f.line_resistance * load_ratio ** 2
            theft = f.ntl_true_pct if (f.theft_start_month >= 0 and m >= f.theft_start_month) else 0.0
            theft = theft * rng.uniform(0.9, 1.1)
            metered = input_mwh * (1 - tech_pct - theft - rng.normal(0.003, 0.004))
            mrows.append({"feeder_id": f.feeder_id, "cell_id": f.cell_id, "region": f.region, "month": m,
                          "input_mwh": input_mwh, "metered_mwh": metered,
                          "tech_loss_est_pct": tech_pct + rng.normal(0, 0.004),
                          "load_ratio": load_ratio, "n_customers": f.n_customers,
                          "ntl_active": int(theft > 0), "tech_loss_true_pct": tech_pct})
    fm = pd.DataFrame(mrows)
    fm["data_origin"] = "synthetic"
    return feeders, fm


def _make_incidents(times, sub, lines, rng):
    rows = []
    for i in range(40):
        t = times[int(rng.integers(0, N_HIST))]
        if rng.random() < 0.5:
            tgt = lines.line_id.iloc[int(rng.integers(0, len(lines)))]
            kind = rng.choice(["line_trip", "overload", "lightning"])
        else:
            tgt = sub.bus_id.iloc[int(rng.integers(0, len(sub)))]
            kind = rng.choice(["transformer_fault", "protection_misoperation", "breaker_failure"])
        rows.append({"time": t, "target": tgt, "kind": kind, "duration_h": float(rng.gamma(1.6, 1.2)),
                     "data_origin": "synthetic"})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
