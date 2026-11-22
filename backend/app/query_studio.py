"""Query Studio: ask anything about the system and get Power BI-style tables and charts.

Flow: question -> planner (OpenAI if configured, otherwise a local keyword planner) -> validated structured QuerySpec ->
pandas executor over a fixed, read-only catalog of datasets built from the service -> chart/table payload -> narrative.

Safety: the LLM never writes code or SQL. It can only emit a JSON spec that is validated against the catalog (known dataset,
known columns, whitelisted operators/aggregations, row limits). Nothing here can modify the system or reach grid control.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import insights as ins
from .security import get_settings
from .service import PowerIQ

Agg = Literal["sum", "mean", "min", "max", "count", "median", "p95", "nunique"]
Op = Literal["=", "!=", ">", ">=", "<", "<=", "in", "contains"]
Bucket = Literal["hour", "day", "week", "month", "hour_of_day", "day_of_week"]
ChartType = Literal["bar", "line", "area", "pie", "scatter", "table"]
MAX_ROWS = 200


class Filter(BaseModel):
    col: str
    op: Op = "="
    value: str | float | int | list[str | float | int]


class Metric(BaseModel):
    col: str = "*"
    agg: Agg = "sum"
    alias: str | None = Field(default=None, max_length=40)


class QuerySpec(BaseModel):
    dataset: str
    title: str = Field(default="", max_length=120)
    filters: list[Filter] = Field(default_factory=list, max_length=6)
    time_bucket: Bucket | None = None
    group_by: list[str] = Field(default_factory=list, max_length=3)
    metrics: list[Metric] = Field(default_factory=list, max_length=6)
    columns: list[str] = Field(default_factory=list, max_length=10)      # raw listing when no metrics
    sort_by: str | None = None
    sort_desc: bool = True
    limit: int = Field(default=50, ge=1, le=MAX_ROWS)
    chart: ChartType = "table"

    @field_validator("filters", mode="before")
    @classmethod
    def _f(cls, v):
        return v or []


# ------------------------------------------------------------------ catalog
class Dataset:
    def __init__(self, name: str, df: pd.DataFrame, description: str, time_col: str | None = None, default_metric: str | None = None):
        self.name, self.df, self.description, self.time_col, self.default_metric = name, df.reset_index(drop=True), description, time_col, default_metric
        self.types: dict[str, str] = {}
        self.values: dict[str, list[str]] = {}
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                self.types[c] = "datetime"
            elif pd.api.types.is_bool_dtype(s):
                self.types[c] = "category"
            elif pd.api.types.is_numeric_dtype(s):
                self.types[c] = "number"
            else:
                self.types[c] = "category"
            if self.types[c] == "category":
                u = s.dropna().astype(str).unique().tolist()
                if len(u) <= 40:
                    self.values[c] = sorted(u)

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description, "rows": int(len(self.df)), "time_column": self.time_col,
                "columns": [{"name": c, "type": t, **({"values": self.values[c]} if c in self.values else {})} for c, t in self.types.items()]}


_cat: dict[int, dict[str, Dataset]] = {}
_lock = threading.Lock()


def _bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
    return df


def build_catalog(s: PowerIQ) -> dict[str, Dataset]:
    with _lock:
        if id(s) in _cat:
            return _cat[id(s)]
    w, n = s.world, s.world.n_hist
    last30 = w.times[n - 720:n]
    names = dict(zip(s.plants.plant_id, s.plants.name))
    src = dict(zip(s.plants.plant_id, s.plants.source))
    D: dict[str, Dataset] = {}

    dh = w.demand.loc[last30].rename_axis("time").reset_index().melt(id_vars="time", var_name="region", value_name="demand_mw")
    dh["period"] = "history"
    df_ = s.dem_fc.rename_axis("time").reset_index().melt(id_vars="time", var_name="region", value_name="demand_mw")
    df_["period"] = "forecast"
    D["demand_hourly"] = Dataset("demand_hourly", pd.concat([dh, df_], ignore_index=True), "Regional electricity demand per hour: last 30 days (history) plus 72 h ML forecast.", "time", "demand_mw")

    sh = w.system.loc[last30].rename_axis("time").reset_index()
    D["system_hourly"] = Dataset("system_hourly", sh, "National system per hour for the last 30 days: demand, supply by source, unserved energy, thermal capacity, hydrology water factor.", "time", "total_demand_mw")

    b = s.baseline
    fs = pd.DataFrame({"time": s.times_fc, "demand_mw": b["demand"], "required_mw": b["required"], "geothermal_mw": b["geo"], "import_mw": b["imp"],
                       "wind_mw": b["wind"], "solar_mw": b["solar"], "hydro_mw": b["hydro"], "thermal_mw": b["thermal"], "curtailed_mw": b["curtailed"],
                       "unserved_mw": b["unserved"], "reserve_margin_pct": b["reserve_margin_pct"]})
    D["forecast_system"] = Dataset("forecast_system", fs, "72 h national dispatch forecast: demand, supply by source, curtailment, unserved energy, reserve margin.", "time", "demand_mw")

    fleet = pd.DataFrame(ins.plant_fleet(s)["plants"])
    D["plants"] = Dataset("plants", fleet, "One row per power station: capacity, capacity factor, availability, 30-day curtailment, unit risk, 72 h forecast output, similarity group.", None, "capacity_mw")

    ph = w.plant_gen.loc[w.times[n - 336:n]].rename_axis("time").reset_index().melt(id_vars="time", var_name="plant_id", value_name="generation_mw")
    ph["plant"] = ph.plant_id.map(names)
    ph["source"] = ph.plant_id.map(src)
    D["plant_hourly"] = Dataset("plant_hourly", ph, "Hourly generation per power station for the last 14 days.", "time", "generation_mw")

    ur = s.maint_risk.copy()
    ur["plant"] = ur.plant_id.map(names)
    ur["drivers"] = ur.drivers.map(lambda x: "; ".join(x) if isinstance(x, (list, tuple)) else str(x))
    D["units_risk"] = Dataset("units_risk", _bool_to_int(ur), "One row per generating unit: 14-day failure probability, sensor readings, days since maintenance, drivers.", None, "risk_probability")

    cf = pd.DataFrame(b["cell_demand"], index=s.times_fc, columns=s.cells.cell_id.values).rename_axis("time").reset_index().melt(id_vars="time", var_name="cell_id", value_name="demand_mw")
    cf["region"] = cf.cell_id.map(dict(zip(s.cells.cell_id, s.cells.region)))
    cf["bus_id"] = cf.cell_id.map(dict(zip(s.cells.cell_id, s.cells.bus_id)))
    D["cell_forecast"] = Dataset("cell_forecast", cf, "Forecast demand per map hexagon (cell) for the next 72 h, with region and nearest grid bus.", "time", "demand_mw")

    lt = s.lb.latest.copy()
    keep = [c for c in ["feeder_id", "cell_id", "region", "month", "n_customers", "input_mwh", "metered_mwh", "tech_loss_est_pct", "ntl_est_pct", "ntl_mwh_est", "tech_mwh_est", "flagged", "anomaly_score", "peer_z"] if c in lt.columns]
    D["feeder_losses"] = Dataset("feeder_losses", _bool_to_int(lt[keep]), "Latest-month energy accounting per distribution feeder: input vs metered energy, estimated technical and non-technical loss, theft flags.", None, "ntl_mwh_est")

    ce = w.curtailment_events.copy()
    ce["plant"] = ce.plant_id.map(names)
    D["curtailment_events"] = Dataset("curtailment_events", ce.drop(columns=["data_origin"], errors="ignore"), "Hourly curtailment events (MW curtailed) per plant with reason, whole history.", "time", "curtailed_mw")

    ll = pd.DataFrame(b["loading"], index=s.times_fc, columns=s.net.lines.line_id.values).rename_axis("time").reset_index().melt(id_vars="time", var_name="line_id", value_name="loading_ratio")
    ll["loading_pct"] = ll.loading_ratio * 100
    lm = s.net.lines.set_index("line_id")
    for c in ("from_bus", "to_bus", "kv", "capacity_mw"):
        ll[c] = ll.line_id.map(lm[c])
    D["line_loading_forecast"] = Dataset("line_loading_forecast", ll.drop(columns=["loading_ratio"]), "72 h forecast loading of each transmission line (DC power-flow approximation).", "time", "loading_pct")

    an = s.anomalies.copy()
    D["anomalies"] = Dataset("anomalies", _bool_to_int(an), "Telemetry anomaly scores for the last 72 h with the driving sensor channel.", "time", "score")
    D["grid_incidents"] = Dataset("grid_incidents", w.grid_incidents.drop(columns=["data_origin"], errors="ignore"), "Historical grid incidents (overloads and other events) with duration.", "time", "duration_h")
    wt = w.weather[w.weather.time.isin(last30)].drop(columns=["data_origin"], errors="ignore")
    D["weather"] = Dataset("weather", wt, "Synthetic hourly weather per region for the last 30 days: temperature, irradiance, wind, cloud, rain.", "time", "temp_c")
    fl = w.faults.copy()
    fl["plant_id"] = fl.unit_id.str.split("-").str[0]
    fl["plant"] = fl.plant_id.map(names)
    fl["source"] = fl.plant_id.map(src)
    D["faults"] = Dataset("faults", fl.drop(columns=["data_origin"], errors="ignore"), "Historical unit faults with date and severity.", "date", "severity")
    with _lock:
        _cat[id(s)] = D
    return D


def catalog_view(s: PowerIQ) -> dict:
    return {"datasets": [d.describe() for d in build_catalog(s).values()], "aggregations": list(Agg.__args__), "operators": list(Op.__args__),
            "time_buckets": list(Bucket.__args__), "charts": list(ChartType.__args__), "max_rows": MAX_ROWS,
            "note": "Read-only catalog built from the synthetic world and model outputs. The planner can only use these datasets and columns."}


# ------------------------------------------------------------------ execution
class QueryError(ValueError):
    pass


def _col(ds: Dataset, c: str) -> str:
    if c not in ds.df.columns:
        raise QueryError(f"Unknown column '{c}' in dataset '{ds.name}'. Available: {', '.join(ds.df.columns)}")
    return c


def _bucket(series: pd.Series, how: str) -> pd.Series:
    t = pd.to_datetime(series)
    if how == "hour":
        return t.dt.floor("h").dt.strftime("%Y-%m-%d %H:00")
    if how == "day":
        return t.dt.strftime("%Y-%m-%d")
    if how == "week":
        return (t.dt.to_period("W").dt.start_time).dt.strftime("%Y-%m-%d")
    if how == "month":
        return t.dt.strftime("%Y-%m")
    if how == "hour_of_day":
        return t.dt.hour
    return t.dt.day_name()


def _clean(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 3)
    if isinstance(v, float):
        return round(v, 3)
    return v if isinstance(v, (int, str, bool)) else str(v)


def execute(s: PowerIQ, spec: QuerySpec) -> dict:
    cat = build_catalog(s)
    if spec.dataset not in cat:
        raise QueryError(f"Unknown dataset '{spec.dataset}'. Available: {', '.join(cat)}")
    ds = cat[spec.dataset]
    df = ds.df
    for f in spec.filters:
        c = _col(ds, f.col)
        t = ds.types[c]
        vals = f.value if isinstance(f.value, list) else [f.value]
        if len(vals) > 20:
            raise QueryError("Too many filter values (max 20).")
        if t == "number":
            try:
                nums = [float(v) for v in vals]
            except (TypeError, ValueError):
                raise QueryError(f"Filter on numeric column '{c}' needs a number.")
            m = {"=": df[c] == nums[0], "!=": df[c] != nums[0], ">": df[c] > nums[0], ">=": df[c] >= nums[0], "<": df[c] < nums[0], "<=": df[c] <= nums[0],
                 "in": df[c].isin(nums), "contains": df[c].astype(str).str.contains(str(vals[0]), regex=False)}[f.op]
        elif t == "datetime":
            try:
                ts = [pd.Timestamp(str(v)) for v in vals]
            except Exception:
                raise QueryError(f"Filter on date column '{c}' needs an ISO date.")
            m = {"=": df[c] == ts[0], "!=": df[c] != ts[0], ">": df[c] > ts[0], ">=": df[c] >= ts[0], "<": df[c] < ts[0], "<=": df[c] <= ts[0],
                 "in": df[c].isin(ts), "contains": df[c].astype(str).str.contains(str(vals[0]), regex=False)}[f.op]
        else:
            sv = [str(v) for v in vals]
            col = df[c].astype(str)
            low = col.str.lower()
            m = {"=": low == sv[0].lower(), "!=": low != sv[0].lower(), "in": low.isin([x.lower() for x in sv]),
                 "contains": low.str.contains(sv[0].lower(), regex=False), ">": col > sv[0], ">=": col >= sv[0], "<": col < sv[0], "<=": col <= sv[0]}[f.op]
        df = df[m]
    if spec.time_bucket and not ds.time_col:
        raise QueryError(f"Dataset '{ds.name}' has no time column, so time_bucket cannot be used.")
    keys: list[str] = []
    work = df.copy()
    if spec.time_bucket:
        work["time_bucket"] = _bucket(work[ds.time_col], spec.time_bucket)
        keys.append("time_bucket")
    for g in spec.group_by:
        keys.append(_col(ds, g))
    metrics = spec.metrics
    if not metrics:
        cols = [_col(ds, c) for c in spec.columns] or list(ds.df.columns)[:8]
        out = work[cols]
        if spec.sort_by:
            out = out.sort_values(_col(ds, spec.sort_by), ascending=not spec.sort_desc)
        total = len(out)
        out = out.head(spec.limit)
    else:
        named: dict[str, tuple[str, Any]] = {}
        for m in metrics:
            alias = (m.alias or (f"{m.agg}_{m.col}" if m.col != "*" else "count")).replace(" ", "_")
            if m.agg == "count":
                named[alias] = (keys[0] if keys else ds.df.columns[0], "size") if m.col == "*" else (_col(ds, m.col), "count")
                continue
            c = _col(ds, m.col)
            if ds.types[c] != "number" and m.agg != "nunique":
                raise QueryError(f"Aggregation '{m.agg}' needs a numeric column; '{c}' is {ds.types[c]}.")
            named[alias] = (c, (lambda x: x.quantile(0.95)) if m.agg == "p95" else m.agg)
        if keys:
            gb = work.groupby(keys, dropna=False)
            out = gb.agg(**{a: pd.NamedAgg(column=c, aggfunc=f) for a, (c, f) in named.items()}).reset_index()
        else:
            row = {}
            for a, (c, f) in named.items():
                row[a] = len(work) if f == "size" else (f(work[c]) if callable(f) else getattr(work[c], f)())
            out = pd.DataFrame([row])
        sort = spec.sort_by or (list(named)[0] if len(out) > 1 and not spec.time_bucket else None)
        if spec.time_bucket and not spec.sort_by:
            out = out.sort_values(keys[0])
        elif sort:
            if sort not in out.columns:
                raise QueryError(f"sort_by '{sort}' is not in the result columns: {', '.join(out.columns)}")
            out = out.sort_values(sort, ascending=not spec.sort_desc)
        total = len(out)
        out = out.tail(spec.limit) if (spec.time_bucket and not spec.sort_by) else out.head(spec.limit)   # trends keep the most recent buckets
    cols = list(out.columns)
    rows = [[_clean(v) for v in r] for r in out.itertuples(index=False, name=None)]
    chart = _chart(spec, cols, rows, keys, list(named) if metrics else [])
    return {"title": spec.title or _auto_title(spec), "dataset": ds.name, "columns": cols, "rows": rows, "row_count": int(total), "truncated": total > len(rows),
            "chart": chart, "spec": spec.model_dump(exclude_none=True)}


def _auto_title(spec: QuerySpec) -> str:
    m = ", ".join(f"{x.agg} {x.col}" for x in spec.metrics) or "rows"
    by = " by " + ", ".join(([spec.time_bucket] if spec.time_bucket else []) + spec.group_by) if (spec.group_by or spec.time_bucket) else ""
    return f"{m} from {spec.dataset}{by}"


def _chart(spec: QuerySpec, cols: list[str], rows: list[list], keys: list[str], metric_cols: list[str]) -> dict:
    t = spec.chart
    if t == "table" or not rows or not metric_cols or not keys:
        return {"type": "table" if (not metric_cols or not keys or not rows) else t, "x": None, "series": [], "data": []}
    ix = {c: i for i, c in enumerate(cols)}
    x = keys[0]
    if len(keys) >= 2 and len(metric_cols) >= 1:     # pivot second key into series (first metric)
        m = metric_cols[0]
        series, data = [], {}
        for r in rows:
            sname = str(r[ix[keys[1]]])
            if sname not in series:
                series.append(sname)
            data.setdefault(r[ix[x]], {})[sname] = r[ix[m]]
        series = series[:8]
        d = [{x: k, **{s_: v.get(s_) for s_ in series}} for k, v in data.items()]
        return {"type": t if t != "pie" else "bar", "x": x, "series": series, "data": d}
    d = [{c: r[ix[c]] for c in [x] + metric_cols} for r in rows]
    return {"type": t, "x": x, "series": metric_cols, "data": d}


# ------------------------------------------------------------------ planners
SYSTEM_PLAN = ("You are the planner for POWERIQ Kenya, an AI electricity-intelligence platform for Kenya's power system (generation, transmission, distribution, "
               "forecasting, maintenance, losses/theft, curtailment, storage, climate risk, digital twin; all data is synthetic). Output ONE JSON object: "
               '{"scope":"in"|"out","mode":"query"|"analysis","queries":[QuerySpec,...],"intent":"short text"}. '
               'scope="out" for anything unrelated to this project (sport, general trivia, coding help, other countries or industries...); then return no queries. '
               'mode="analysis" when the user asks why/what should we do/recommend/assess/summarise/prioritise; you may still add up to 4 queries to ground it. '
               "QuerySpec fields: dataset, title, filters[{col,op,value}], time_bucket (hour|day|week|month|hour_of_day|day_of_week, only when the dataset has a time column), "
               "group_by[columns], metrics[{col,agg,alias}] (agg: sum|mean|min|max|count|median|p95|nunique; col '*' for count), columns[] (raw listing when no metrics), "
               "sort_by, sort_desc, limit (<=200), chart (bar|line|area|pie|scatter|table). Use ONLY datasets and columns in the catalog and categorical values exactly as listed. "
               "Prefer time_bucket+line for trends, group_by+bar for comparisons, pie for shares of a few categories. Never output code, SQL, URLs or text outside the JSON.")

SYSTEM_NARRATE = ("You are the POWERIQ Kenya analyst. Answer ONLY from the evidence JSON (query results, computed statistics, model findings). Start with a direct 2-4 sentence "
                  "answer, then up to 4 bullets of analysis: comparisons, trends, outliers, concentration, likely drivers only where the evidence supports them, and caveats "
                  "(synthetic data, truncated rows, small samples). Use exact numbers. Never invent values, regulations or events. Recommendations are advisory and need "
                  "human approval; POWERIQ never controls the grid.")

def _catalog_prompt(s: PowerIQ) -> str:
    out = []
    for d in build_catalog(s).values():
        cols = []
        for c, t in d.types.items():
            cols.append(f"{c}:{t}" + (f"[{'|'.join(d.values[c][:25])}]" if c in d.values else ""))
        out.append(f"- {d.name} ({d.description}) time={d.time_col} cols: " + ", ".join(cols))
    return "\n".join(out)


def _chat(messages: list[dict], json_mode: bool) -> tuple[str | None, str | None]:
    st = get_settings()
    if not st.openai_api_key:
        return None, "OPENAI_API_KEY not configured"
    body: dict[str, Any] = {"model": st.openai_model, "temperature": 0.1, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        r = httpx.post("https://api.openai.com/v1/chat/completions", timeout=40, headers={"Authorization": f"Bearer {st.openai_api_key}"}, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:160]}"


class Plan(BaseModel):
    scope: str = "in"
    mode: str = "query"
    specs: list[QuerySpec] = Field(default_factory=list)
    intent: str = ""


def llm_plan(s: PowerIQ, question: str, history: list[dict] | None) -> tuple[Plan | None, str | None]:
    msgs = [{"role": "system", "content": SYSTEM_PLAN + "\n\nCATALOG:\n" + _catalog_prompt(s)}]
    for h in (history or [])[-4:]:
        msgs.append({"role": "user", "content": str(h.get("question", ""))[:300]})
    msgs.append({"role": "user", "content": question})
    text, err = _chat(msgs, True)
    if text is None:
        return None, err
    try:
        j = json.loads(text)
        scope = "out" if str(j.get("scope", "in")).lower() == "out" else "in"
        mode = "analysis" if str(j.get("mode", "query")).lower() == "analysis" else "query"
        specs = [QuerySpec.model_validate(q) for q in (j.get("queries") or [])[:4]]
        return Plan(scope=scope, mode=mode, specs=specs, intent=str(j.get("intent", ""))[:300]), None
    except (json.JSONDecodeError, ValidationError, AttributeError, TypeError) as e:
        return None, f"planner returned an invalid spec ({type(e).__name__})"


SCOPE_RE = re.compile(r"\b(electric\w*|power|energy|grid|kenya\w*|kplc|kengen|ketraco|epra|plants?|stations?|geotherm\w*|hydro\w*|wind|solar|thermal|demand|load|generat\w*|supply|reserve|"
                      r"curtail\w*|loss(?:es)?|theft|meters?|metering|feeders?|substations?|transmission|voltage|transformers?|batter\w*|storage|maintenance|faults?|outages?|blackouts?|"
                      r"tariffs?|renewables?|dispatch|regions?|hex\w*|cells?|anomal\w*|capacity|megawatts?|mw|mwh|kwh|twin|scenarios?|climate|drought|rain\w*|weather|temperature|"
                      r"poweriq|platform|audit|approvals?|agents?|copilot|olkaria|turkana|gitaru|kindaruma|embakasi|nairobi|mombasa|kisumu|nakuru|import\w*|unserved|interruptions?|"
                      r"infrastructure|vandalism|equipment|turbines?|generators?|reservoirs?|stability|overloads?|loading|kenyans?)\b", re.I)
META_RE = re.compile(r"\b(what can you do|what can i ask|help|capabilit\w*|what data|which datasets|list datasets|what do you know|how do you work)\b", re.I)
ANALYSIS_RE = re.compile(r"\b(why|explain|recommend\w*|should|improve|priorit\w*|summari[sz]e|overall|assess\w*|biggest|main problems?|root cause|what can|how can|insights?|analy[sz]e|analysis|risks?)\b", re.I)


def in_scope(s: PowerIQ, question: str) -> bool:
    if SCOPE_RE.search(question) or any(any(k in question.lower() for k in kws) for _, kws in KEYWORDS):
        return True
    ql = question.lower()
    for ds in build_catalog(s).values():
        for vals in ds.values.values():
            if any(len(v) >= 4 and v.lower() in ql for v in vals):
                return True
    return False


def capabilities_text(s: PowerIQ) -> str:
    cat = build_catalog(s)
    return ("I can answer questions about Kenya's power system as modelled in POWERIQ (synthetic data): " + "; ".join(f"{d.name} ({d.description.split(':')[0].split('.')[0].lower()})" for d in cat.values()) +
            ". Ask for tables, trends, comparisons or analysis, e.g. 'Why could reserve margin fall this week and what should we do?' or 'Compare curtailment by plant'. "
            "Questions outside this project (sport, general knowledge, other countries) are not answered here.")


KEYWORDS: list[tuple[str, list[str]]] = [
    ("curtailment_events", ["curtail"]), ("feeder_losses", ["loss", "theft", "ntl", "feeder", "non-technical", "non technical", "meter"]),
    ("line_loading_forecast", ["line", "loading", "overload", "transmission"]), ("units_risk", ["risk", "maintenance", "failure", "unit", "sensor", "vibration"]),
    ("faults", ["fault history", "faults"]), ("anomalies", ["anomal", "telemetry"]), ("grid_incidents", ["incident"]),
    ("weather", ["weather", "temperature", "rain", "irradiance", "cloud", "wind speed"]), ("plant_hourly", ["generation by", "output of", "generated", "plant output", "station output"]),
    ("plants", ["plant", "station", "capacity", "fleet", "geothermal", "hydro"]), ("cell_forecast", ["cell", "hex", "bus"]),
    ("forecast_system", ["forecast", "next 24", "next 72", "tomorrow", "reserve", "unserved", "dispatch", "shortfall", "mix"]),
    ("demand_hourly", ["demand", "load", "consumption", "usage"]), ("system_hourly", ["system", "national", "history", "historical", "supply", "water"]),
]
GROUP_WORDS = {"region": "region", "regions": "region", "source": "source", "sources": "source", "plant": "plant", "plants": "plant", "station": "plant", "stations": "plant",
               "reason": "reason", "kind": "kind", "line": "line_id", "lines": "line_id", "feeder": "feeder_id", "feeders": "feeder_id", "cell": "cell_id", "unit": "unit_id",
               "bus": "bus_id", "channel": "driver_channel", "asset": "asset_id", "cluster": "cluster", "group": "cluster", "month": "month"}
TIME_WORDS = [("hour of day", "hour_of_day"), ("day of week", "day_of_week"), ("hourly", "hour"), ("per hour", "hour"), ("by hour", "hour"), ("each hour", "hour"),
              ("daily", "day"), ("per day", "day"), ("by day", "day"), ("each day", "day"), ("weekly", "week"), ("by week", "week"), ("monthly", "month")]


def _pick_metric(ds: Dataset, ql: str) -> str:
    cols = [c for c, t in ds.types.items() if t == "number"]
    words = re.findall(r"[a-z]+", ql)
    best, score = None, 0
    for c in cols:
        parts = [p for p in c.lower().split("_") if p not in ("mw", "mwh", "pct", "est") and len(p) > 2]
        sc = sum(1 for p in parts if p in words or any(w.startswith(p[:5]) for w in words if len(w) > 3))
        if sc > score:
            best, score = c, sc
    return best or ds.default_metric or cols[0]


def local_plan(s: PowerIQ, question: str, strict: bool = False) -> tuple[list[QuerySpec], str]:
    ql = question.lower()
    cat = build_catalog(s)
    name = next((n for n, kws in KEYWORDS if any(k in ql for k in kws)), None)
    if name is None and strict:
        return [], "No specific dataset matched."
    ds = cat[name or "forecast_system"]
    metric = _pick_metric(ds, ql)
    agg: Agg = "mean"
    explicit = True
    if re.search(r"\b(total|sum)\b|how much", ql):
        agg = "sum"
    elif re.search(r"\b(max|maximum|peak|highest|largest|worst|most)\b", ql):
        agg = "max"
    elif re.search(r"\b(min|minimum|lowest|smallest|least)\b", ql):
        agg = "min"
    elif re.search(r"\b(average|mean|avg|typical)\b", ql):
        agg = "mean"
    elif metric.endswith("_mwh") or metric.startswith("capacity") or "mwh" in metric:
        agg = "sum"
    else:
        explicit = False
    count_q = re.search(r"how many|number of|\bcount\b", ql) is not None
    tb = next((b for k, b in TIME_WORDS if k in ql), None)
    if tb and not ds.time_col:
        tb = None
    group: list[str] = []
    for m in re.finditer(r"\b(?:by|per|each|across|for each|breakdown by)\s+([a-z ]{3,20}?)(?=\s+(?:and|in|for|on|during|over|with|where)\b|[,.?]|$)", ql):
        for w_ in m.group(1).split():
            c = GROUP_WORDS.get(w_)
            if c and c in ds.df.columns and c not in group:
                group.append(c)
    filters = []
    for c, vals in ds.values.items():
        if c in ("drivers", "month"):
            continue
        for v in vals:
            if len(v) >= 3 and re.search(r"(?<![a-z0-9_])" + re.escape(v.lower().replace("_", " ")) + r"(?![a-z0-9_])", ql.replace("_", " ")):
                filters.append(Filter(col=c, op="=", value=v))
                break
    ENT = [(r"\b(stations?|plants?)\b", "plant"), (r"\bunits?\b", "unit_id"), (r"\blines?\b", "line_id"), (r"\bfeeders?\b", "feeder_id"),
           (r"\bregions?\b", "region"), (r"\bcells?\b", "cell_id")]
    if not group and not tb and re.search(r"\b(which|top|highest|lowest|rank|worst|most|compare|biggest|largest)\b", ql):
        for pat, col in ENT:
            col = "name" if (ds.name == "plants" and col == "plant") else col
            if re.search(pat, ql) and col in ds.df.columns and col not in {f.col for f in filters}:
                group = [col]
                break
    # a value used as a filter is not also a group-by
    group = [g for g in group if g not in {f.col for f in filters}][:2]
    if not group and not tb and not filters and re.search(r"\b(compare|breakdown|share|split|distribution|rank|top|which|where)\b", ql):
        for c in ("region", "source", "plant", "reason", "kind", "line_id", "feeder_id", "asset_type", "region"):
            if c in ds.df.columns:
                group = [c]
                break
    topn = re.search(r"top\s+(\d{1,3})", ql)
    limit = int(topn.group(1)) if topn else (24 if tb == "hour" else 20)
    if tb == "hour" and not re.search(r"\bnext\b|\blast\b|\bforecast\b", ql) and ds.name in ("demand_hourly", "system_hourly", "weather", "plant_hourly"):
        limit = 200
    metrics = [Metric(col="*", agg="count", alias="count")] if count_q and not (tb and False) else [Metric(col=metric, agg=agg, alias=f"{agg}_{metric}")]
    if tb == "hour" and not explicit and ds.name in ("plant_hourly", "demand_hourly", "cell_forecast", "curtailment_events"):
        agg = "sum"   # several entities share each hour: the hourly total is the meaningful number
        metrics = [Metric(col=metric, agg="sum", alias=f"total_{metric}")]
    if ds.name == "forecast_system" and not group and not tb:
        tb = "hour"
    if ds.name == "forecast_system" and tb and metric == "demand_mw" and re.search(r"mix|supply|source|generation", ql):
        metrics = [Metric(col=c, agg="sum", alias=c) for c in ("geothermal_mw", "hydro_mw", "wind_mw", "solar_mw", "thermal_mw", "import_mw")]
    chart: ChartType = "table"
    if tb:
        chart = "area" if len(metrics) > 2 else "line"
    elif group:
        chart = "pie" if re.search(r"share|split|proportion|pie", ql) and len(group) == 1 else "bar"
    if group and not tb and not explicit and not count_q and agg == "mean" and re.search(r"\b(highest|worst|most|top|largest|biggest)\b", ql):
        metrics = [Metric(col=metric, agg="max", alias=f"max_{metric}")]
    spec = QuerySpec(dataset=ds.name, filters=filters, time_bucket=tb, group_by=group, metrics=metrics, sort_by=None, sort_desc=not re.search(r"lowest|least|min", ql),
                     limit=min(limit, MAX_ROWS), chart=chart)
    if filters or group or tb:
        spec.title = _auto_title(spec) + ("" if not filters else " (" + ", ".join(f"{f.col}={f.value}" for f in filters) + ")")
    return [spec], f"Local planner matched dataset '{ds.name}'."


def follow_ups(spec: QuerySpec) -> list[str]:
    d = spec.dataset
    base = {"forecast_system": ["What is the lowest reserve margin in the next 72 hours?", "Show the supply mix by hour"],
            "plants": ["Top 5 stations by curtailed energy", "Total capacity by source"],
            "feeder_losses": ["Total non-technical loss by region", "Top 10 feeders by non-technical loss"],
            "units_risk": ["Top 10 units by failure risk", "Average failure risk by source"],
            "curtailment_events": ["Total curtailed energy by plant", "Curtailment by hour of day"],
            "demand_hourly": ["Average demand by hour of day", "Total demand by region"],
            "line_loading_forecast": ["Max loading by line", "Which lines exceed 80% loading?"]}
    return base.get(d, ["Show the supply mix by hour", "Total capacity by source"])


def run_specs(s: PowerIQ, specs: list[QuerySpec]) -> tuple[list[dict], list[str]]:
    res, errs = [], []
    for sp in specs:
        try:
            res.append(execute(s, sp))
        except QueryError as e:
            errs.append(str(e))
    return res, errs


def _evidence(results: list[dict]) -> list[dict]:
    return [{"title": r["title"], "columns": r["columns"], "rows": r["rows"][:40], "row_count": r["row_count"], "truncated": r["truncated"]} for r in results]


def analyze(results: list[dict]) -> list[dict]:
    """Descriptive statistics per result: spread, concentration, trend, outliers and correlation. Pure computation, no model."""
    out = []
    for r in results:
        cols, rows = r["columns"], r["rows"]
        if not rows:
            continue
        spec = r["spec"]
        keys = list(spec.get("group_by") or []) + (["time_bucket"] if spec.get("time_bucket") else [])
        keys = [k for k in keys if k in cols]
        metrics = [c for i, c in enumerate(cols) if c not in keys and any(isinstance(x[i], (int, float)) and not isinstance(x[i], bool) for x in rows)]
        ix = {c: i for i, c in enumerate(cols)}
        label = (lambda row: str(row[ix[keys[0]]])) if keys else (lambda row: "all")
        item: dict = {"title": r["title"], "stats": {}, "notes": []}
        for c in metrics[:4]:
            pairs = [(label(x), float(x[ix[c]])) for x in rows if isinstance(x[ix[c]], (int, float))]
            if not pairs:
                continue
            v = np.array([p[1] for p in pairs])
            item["stats"][c] = {"n": int(len(v)), "mean": round(float(v.mean()), 3), "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
                                "std": round(float(v.std()), 3), "total": round(float(v.sum()), 3)}
            if len(v) < 2 or not keys:
                continue
            hi, lo = pairs[int(v.argmax())], pairs[int(v.argmin())]
            if spec.get("time_bucket"):
                slope = float(np.polyfit(np.arange(len(v)), v, 1)[0])
                first, last = float(v[0]), float(v[-1])
                chg = (last - first) / abs(first) * 100 if abs(first) > 1e-9 else None
                item["notes"].append(f"{c}: {'rising' if slope > 0 else 'falling' if slope < 0 else 'flat'} across {len(v)} buckets"
                                     + (f" ({chg:+.1f}% first to last)" if chg is not None else "") + f"; peak {hi[1]:,.3g} at {hi[0]}, low {lo[1]:,.3g} at {lo[0]}.")
            else:
                tot = float(v.sum())
                if tot > 0 and (v >= 0).all() and len(v) >= 3:
                    item["notes"].append(f"{c}: {hi[0]} is {hi[1] / tot * 100:.0f}% of the total; the top 3 hold {np.sort(v)[::-1][:3].sum() / tot * 100:.0f}%.")
                if len(v) >= 6 and v.std() > 0:
                    z = (v - v.mean()) / v.std()
                    out_ = [pairs[i][0] for i in np.where(np.abs(z) > 2.5)[0]][:3]
                    if out_:
                        item["notes"].append(f"{c}: unusually far from the mean (>2.5 sd): {', '.join(out_)}.")
        if len(metrics) >= 2 and len(rows) >= 5:
            a_, b_ = [x[ix[metrics[0]]] for x in rows], [x[ix[metrics[1]]] for x in rows]
            ok = [(p, q) for p, q in zip(a_, b_) if isinstance(p, (int, float)) and isinstance(q, (int, float))]
            if len(ok) >= 5 and np.std([p for p, _ in ok]) > 0 and np.std([q for _, q in ok]) > 0:
                cc = float(np.corrcoef([p for p, _ in ok], [q for _, q in ok])[0, 1])
                item["notes"].append(f"Correlation between {metrics[0]} and {metrics[1]}: {cc:+.2f} (association only, not causation).")
        out.append(item)
    return out


def _local_narrative(question: str, results: list[dict], errs: list[str], intent: str, analysis: list[dict] | None = None) -> str:
    if not results:
        return "No query could be run. " + (" ".join(errs) if errs else intent or "Try naming a dataset such as demand, plants, losses, curtailment, line loading or reserve margin.")
    lines = []
    for r in results:
        if r["row_count"] == 0:
            lines.append(f"{r['title']}: no rows matched.")
            continue
        cols, rows = r["columns"], r["rows"]
        num = [i for i, c in enumerate(cols) if isinstance(rows[0][i], (int, float)) and not isinstance(rows[0][i], bool)]
        if len(rows) == 1 and num:
            lines.append(f"{r['title']}: " + ", ".join(f"{cols[i]} = {rows[0][i]:,}" for i in num[:4]) + ".")
        elif num:
            i = num[-1]
            hi, lo = max(rows, key=lambda x: x[i] if x[i] is not None else -1e18), min(rows, key=lambda x: x[i] if x[i] is not None else 1e18)
            lab = lambda x: str(x[0])
            lines.append(f"{r['title']}: {r['row_count']} rows; highest {cols[i]} {hi[i]:,} at {lab(hi)}, lowest {lo[i]:,} at {lab(lo)}.")
        else:
            lines.append(f"{r['title']}: {r['row_count']} rows returned.")
    notes = [n for a in (analysis or []) for n in a["notes"]][:4]
    return "\n".join(f"- {x}" for x in lines + notes) + "\n\nComputed by the local planner (no LLM); data is synthetic."


def _refusal(s: PowerIQ, question: str, mode: str, status: str, kind: str) -> dict:
    text = ("That question is outside the scope of POWERIQ Kenya, so I haven't run anything. " if kind == "out_of_scope" else "") + capabilities_text(s)
    return {"data_origin": "synthetic+model", "mode": mode, "llm_status": status, "question": question, "kind": kind, "intent": kind, "results": [], "errors": [], "analysis": [],
            "findings": [], "narrative": text, "follow_ups": ["Why could the reserve margin fall in the next 72 hours and what should we do?", "Compare curtailment by plant",
                                                              "Which regions have the highest non-technical loss?"], "notice": "Scope-limited: only questions about this project are answered."}


def status() -> dict:
    st = get_settings()
    return {"openai_configured": bool(st.openai_api_key), "model": st.openai_model,
            "how_to_enable": "Set OPENAI_API_KEY (and optionally POWERIQ_OPENAI_MODEL) in the server environment or .env and restart. The key is never sent to the browser.",
            "tested_against_real_api": False}


def test_openai() -> dict:
    text, err = _chat([{"role": "user", "content": "Reply with the single word: ok"}], False)
    return {"ok": text is not None, "reply": (text or "")[:40], "error": err, **status()}


def ask(s: PowerIQ, question: str, history: list[dict] | None = None) -> dict:
    plan, err = llm_plan(s, question, history)
    mode = "openai" if plan is not None else "local-planner"
    llm_status = "ok" if plan is not None else f"not used: {err}"
    if plan is None:
        if META_RE.search(question) and not SCOPE_RE.search(question.replace("help", "")):
            return _refusal(s, question, mode, llm_status, "capabilities")
        if not in_scope(s, question):
            return _refusal(s, question, mode, llm_status, "out_of_scope")
        analysis_mode = ANALYSIS_RE.search(question) is not None
        specs, intent = local_plan(s, question, strict=analysis_mode)
        clauses = [c.strip() for c in re.split(r"\b(?:and then|and also|also|then|plus)\b|;|\band\b", question, flags=re.I) if len(c.strip()) > 3]
        if len(clauses) > 1:       # complex, multi-part question: plan each clause that names its own dataset
            multi: list[QuerySpec] = []
            for c in clauses[:4]:
                for sp in local_plan(s, c, strict=True)[0]:
                    if all((sp.dataset, sp.time_bucket, tuple(sp.group_by)) != (y.dataset, y.time_bucket, tuple(y.group_by)) for y in multi):
                        multi.append(sp)
            if len(multi) >= 2:
                specs, intent = multi[:3], f"Local planner split the question into {len(multi[:3])} parts."
        plan = Plan(scope="in", mode="analysis" if analysis_mode else "query", specs=specs, intent=intent)
    elif plan.scope == "out":
        return _refusal(s, question, mode, llm_status, "out_of_scope")
    specs, intent = plan.specs, plan.intent
    results, errs = run_specs(s, specs)
    stats = analyze(results)
    findings = ins.findings(s) if plan.mode == "analysis" else []
    narrative, nerr = None, None
    if mode == "openai" and (results or findings):
        ev = {"results": _evidence(results), "statistics": stats}
        if findings:
            ev["model_findings"] = [{k: f[k] for k in ("area", "severity", "finding", "action")} for f in findings]
            ev["scorecard"] = ins.scorecard(s)["rows"]
        narrative, nerr = _chat([{"role": "system", "content": SYSTEM_NARRATE}, {"role": "user", "content": f"Question: {question}\n\nEvidence JSON:\n{json.dumps(ev, default=str)[:24000]}"}], False)
        if nerr:
            llm_status = f"planner ok; narrative not used: {nerr}"
    if not narrative:
        if findings:
            narrative = ins.ai_analysis(s, question)["narrative"]
            if results:
                narrative += "\n\nQuery results:\n" + _local_narrative(question, results, errs, intent or "", stats)
        else:
            narrative = _local_narrative(question, results, errs, intent or "", stats)
    return {"data_origin": "synthetic+model", "mode": mode, "llm_status": llm_status, "question": question, "kind": plan.mode, "intent": intent, "results": results, "errors": errs,
            "analysis": stats, "findings": [{k: f[k] for k in ("id", "area", "severity", "finding", "action")} for f in findings], "narrative": narrative,
            "follow_ups": follow_ups(specs[0]) if specs else ["Which regions have the highest non-technical loss?", "Show the supply mix by hour"],
            "notice": "Queries are read-only over a fixed catalog. The LLM only proposes a validated spec; it never runs code. Advisory only."}


DEFAULT_DASHBOARD = [
    QuerySpec(dataset="forecast_system", title="Demand vs required generation, next 72 h (MW)", time_bucket="hour", metrics=[Metric(col="demand_mw", agg="sum", alias="demand_mw"), Metric(col="required_mw", agg="sum", alias="required_mw")], chart="line", limit=100),
    QuerySpec(dataset="forecast_system", title="Supply mix by hour (MW)", time_bucket="hour", metrics=[Metric(col=c, agg="sum", alias=c) for c in ("geothermal_mw", "hydro_mw", "wind_mw", "solar_mw", "thermal_mw", "import_mw")], chart="area", limit=100),
    QuerySpec(dataset="plants", title="Installed capacity by source (MW)", group_by=["source"], metrics=[Metric(col="capacity_mw", agg="sum", alias="capacity_mw")], chart="pie"),
    QuerySpec(dataset="curtailment_events", title="Most curtailed stations (MW-hours)", group_by=["plant"], metrics=[Metric(col="curtailed_mw", agg="sum", alias="curtailed_mwh")], sort_by="curtailed_mwh", limit=8, chart="bar"),
    QuerySpec(dataset="feeder_losses", title="Estimated non-technical loss by region (MWh)", group_by=["region"], metrics=[Metric(col="ntl_mwh_est", agg="sum", alias="ntl_mwh")], sort_by="ntl_mwh", chart="bar"),
    QuerySpec(dataset="units_risk", title="Highest unit failure risk by station", group_by=["plant"], metrics=[Metric(col="risk_probability", agg="max", alias="max_risk")], sort_by="max_risk", limit=8, chart="bar"),
]


def dashboard(s: PowerIQ, goal: str | None) -> dict:
    specs, intent, err, mode = None, None, None, "local-default"
    if goal and goal.strip():
        plan, err = llm_plan(s, "Design a dashboard of 4-6 tiles for: " + goal.strip(), None)
        if plan and plan.specs and plan.scope == "in":
            specs, mode = plan.specs, "openai"
    if not specs:
        specs = DEFAULT_DASHBOARD
    results, errs = run_specs(s, specs)
    return {"data_origin": "synthetic+model", "mode": mode, "llm_status": "ok" if mode == "openai" else (f"not used: {err}" if goal else "default dashboard"),
            "goal": goal, "tiles": results, "errors": errs,
            "notice": "Power BI itself is not connected in this build; use the CSV export on any tile to import the table into Power BI."}
