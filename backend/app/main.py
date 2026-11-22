"""POWERIQ Kenya API (FastAPI). All /api/v1 routes except /health require authentication."""
from __future__ import annotations

import math
import os
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import copilot as copilot_mod
from . import fault_hub as fh
from . import insights as ins
from . import ml
from . import query_studio as qs
from .agents import AgentInput, AgentRegistry
from .rag import get_kb
from .security import AuditLog, Principal, get_audit, get_settings, require
from .service import PowerIQ, get_service

API = "/api/v1"
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    state["audit"] = get_audit()
    state["svc"] = get_service()
    state["agents"] = AgentRegistry(state["svc"])
    yield


app = FastAPI(title="POWERIQ Kenya API", version="1.0.0", lifespan=lifespan,
              description="AI electricity intelligence platform (decision support only; synthetic data; never connected to live grid control).")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("POWERIQ_CORS", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])


def err(code: str, message: str, status: int, request: Request, details: Any = None, headers: dict | None = None):
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"code": code, "message": message, "details": details,
                                           "request_id": getattr(request.state, "request_id", uuid.uuid4().hex[:12])}})


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = request.state.request_id
    return resp


@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    d = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    return err(d.get("code", "http_error"), d.get("message", ""), exc.status_code, request, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def val_exc(request: Request, exc: RequestValidationError):
    return err("validation_error", "Request validation failed.", 422, request,
               [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()])


@app.exception_handler(ValueError)
async def value_exc(request: Request, exc: ValueError):
    return err("invalid_request", str(exc), 400, request)


@app.exception_handler(Exception)
async def any_exc(request: Request, exc: Exception):
    return err("internal_error", "Unexpected server error.", 500, request, {"type": type(exc).__name__})


def svc() -> PowerIQ:
    return state["svc"]


def agents() -> AgentRegistry:
    return state["agents"]


def audit() -> AuditLog:
    return state["audit"]


R = Depends(require("viewer"))
A = Depends(require("analyst"))
AP = Depends(require("approver"))


# ------------------------------------------------------------------ health & meta
@app.get("/health", tags=["meta"])
def health():
    ready = "svc" in state
    return {"status": "ok" if ready else "starting", "ready": ready, "service": "poweriq-kenya"}


@app.get(f"{API}/integrations/status", tags=["meta"])
def integrations(p: Principal = R):
    st = get_settings()
    return {"integrations": [
        {"name": "OpenAI API (copilot narrative)", "status": "configured_untested" if st.openai_api_key else "not_configured",
         "note": "Set OPENAI_API_KEY on the backend. Falls back to a local template narrative on any failure."},
        {"name": "Microsoft Foundry IQ", "status": "not_configured", "note": "Local hybrid retriever is used instead."},
        {"name": "Fabric IQ / OneLake / Eventstream", "status": "not_configured", "note": "No connection has been made or tested."},
        {"name": "Azure AI Search", "status": "not_configured", "note": "Local hybrid retriever is used instead."},
        {"name": "Open-Meteo / NASA POWER", "status": "not_enabled", "note": "Weather is synthetic."},
        {"name": "Hedera testnet anchoring", "status": "not_implemented", "note": "Local hash-chained audit log only."},
        {"name": "MCP server", "status": "implemented_basic", "note": f"POST {API}/mcp (JSON-RPC: initialize, tools/list, tools/call). Read-only tools only; not tested with external MCP clients."},
        {"name": "Live grid / SCADA", "status": "never_connected", "note": "By design."}],
        "environment": st.env, "knowledge_base": {"retriever": "local hybrid", "documents": len(get_kb().documents())}}


@app.get(f"{API}/overview", tags=["analytics"])
def overview(p: Principal = R):
    return svc().overview()


@app.get(f"{API}/metrics", tags=["analytics"])
def metrics(p: Principal = R):
    return svc().metrics_summary()


@app.get(f"{API}/generation", tags=["analytics"])
def generation(p: Principal = R):
    return svc().generation_view()


@app.get(f"{API}/demand", tags=["analytics"])
def demand(p: Principal = R):
    return svc().demand_view()


@app.get(f"{API}/forecast", tags=["analytics"])
def forecast(horizon_hours: int = Query(24), p: Principal = R):
    if horizon_hours not in (1, 6, 24, 72):
        raise HTTPException(422, detail={"code": "validation_error", "message": "horizon_hours must be one of 1, 6, 24, 72."})
    d, g = svc().demand_view(), svc().generation_view()
    n = horizon_hours
    return {"horizon_hours": n, "time": d["time"][:n], "demand_mw": d["national_mw"][:n], "demand_lo80": d["lo80_mw"][:n], "demand_hi80": d["hi80_mw"][:n],
            "generation_mw": {k: v[:n] for k, v in g["mean_mw"].items()}, "model_version": d["model_version"], "data_origin": "forecast",
            "holdout_metrics": {"demand": d["metrics"][f"h{n}"]["national"], "generation": {k: v for k, v in g["metrics"].items() if k.endswith(f"_h{n}")}}}


@app.get(f"{API}/maintenance", tags=["analytics"])
def maintenance(p: Principal = R):
    return svc().maintenance_view()


@app.get(f"{API}/anomalies", tags=["analytics"])
def anomalies(p: Principal = R):
    return svc().anomalies_view()


@app.get(f"{API}/losses", tags=["analytics"])
def losses(p: Principal = R):
    return svc().losses_view()


@app.get(f"{API}/curtailment", tags=["analytics"])
def curtailment(p: Principal = R):
    return svc().curtailment_view()


@app.get(f"{API}/grid", tags=["analytics"])
def grid(hour: int = Query(0, ge=0, le=71), p: Principal = R):
    return svc().grid_view(hour)


@app.get(f"{API}/grid/n1", tags=["analytics"])
def grid_n1(hour: int | None = Query(None, ge=0, le=71), p: Principal = R):
    return svc().n1_screen(hour)


@app.get(f"{API}/climate", tags=["analytics"])
def climate(p: Principal = R):
    return svc().climate_view()


@app.get(f"{API}/usage", tags=["analytics"])
def usage(p: Principal = R):
    return svc().usage_view()


# ------------------------------------------------------------------ Kenya problem insights (additive)
@app.get(f"{API}/insights/scorecard", tags=["insights"])
def insights_scorecard(p: Principal = R):
    return ins.scorecard(svc())


@app.get(f"{API}/insights/reserve-risk", tags=["insights"])
def insights_reserve(p: Principal = R):
    return ins.reserve_risk(svc())


@app.get(f"{API}/insights/curtailment-storage", tags=["insights"])
def insights_curtailment(p: Principal = R):
    return ins.curtailment_storage(svc())


@app.get(f"{API}/insights/climate-stress", tags=["insights"])
def insights_climate(p: Principal = R):
    return ins.climate_stress(svc())


@app.get(f"{API}/insights/theft-planner", tags=["insights"])
def insights_theft(p: Principal = R):
    return ins.theft_planner(svc())


@app.get(f"{API}/insights/plant-fleet", tags=["insights"])
def insights_fleet(p: Principal = R):
    return ins.plant_fleet(svc())


@app.get(f"{API}/insights/demand-structure", tags=["insights"])
def insights_demand(p: Principal = R):
    return ins.demand_structure(svc())


@app.get(f"{API}/insights/analysis", tags=["insights"])
def insights_analysis(question: str = Query("Summarise the main reliability and efficiency problems in Kenya's power system and what to do first.", max_length=500), p: Principal = R):
    out = ins.ai_analysis(svc(), question)
    audit().append("insights_analysis", p.key_id, p.role, {"mode": out["mode"], "findings": [f["id"] for f in out["findings"]]})
    return out


class ProposeFinding(BaseModel):
    finding_id: str = Field(pattern=r"^F-[A-Z]{2,10}$")


@app.post(f"{API}/insights/propose", tags=["insights"])
def insights_propose(b: ProposeFinding, p: Principal = Depends(require("analyst"))):
    f = next((x for x in ins.findings(svc()) if x["id"] == b.finding_id), None)
    if f is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "No such finding in the current analysis."})
    rid = audit().add_recommendation("insights", f"[{f['area']}] {f['action']}", p.key_id, {"finding": f["finding"], "finding_id": f["id"]})
    audit().append("recommendation_created", p.key_id, p.role, {"id": rid, "finding_id": f["id"]})
    return {"recommendation_id": rid, "status": "pending", "notice": "Sent to the approval queue. POWERIQ cannot execute operational actions."}


# ------------------------------------------------------------------ Query Studio (ask anything; additive)
class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=6)


class DashBody(BaseModel):
    goal: str | None = Field(default=None, max_length=300)


@app.get(f"{API}/query/status", tags=["query"])
def query_status(p: Principal = R):
    return qs.status()


@app.post(f"{API}/query/test-openai", tags=["query"])
def query_test_openai(p: Principal = Depends(require("analyst"))):
    out = qs.test_openai()
    audit().append("openai_test", p.key_id, p.role, {"ok": out["ok"]})
    return out


@app.get(f"{API}/query/catalog", tags=["query"])
def query_catalog(p: Principal = R):
    return qs.catalog_view(svc())


@app.post(f"{API}/query/run", tags=["query"])
def query_run(spec: qs.QuerySpec, p: Principal = R):
    try:
        out = qs.execute(svc(), spec)
    except qs.QueryError as e:
        raise HTTPException(422, detail={"code": "invalid_query", "message": str(e)})
    audit().append("query_run", p.key_id, p.role, {"dataset": spec.dataset, "metrics": [m.agg for m in spec.metrics]})
    return out


@app.post(f"{API}/query/ask", tags=["query"])
def query_ask(b: AskBody, p: Principal = R):
    out = qs.ask(svc(), b.question, b.history)
    audit().append("query_ask", p.key_id, p.role, {"question": b.question[:200], "mode": out["mode"], "datasets": [r["dataset"] for r in out["results"]], "errors": len(out["errors"])})
    return out


@app.post(f"{API}/query/dashboard", tags=["query"])
def query_dashboard(b: DashBody, p: Principal = R):
    out = qs.dashboard(svc(), b.goal)
    audit().append("query_dashboard", p.key_id, p.role, {"goal": (b.goal or "")[:200], "mode": out["mode"], "tiles": len(out["tiles"])})
    return out


# ------------------------------------------------------------------ Fault Hub (additive)
class FaultReport(BaseModel):
    asset_type: Literal["plant", "substation", "line", "other"]
    asset_id: str | None = Field(default=None, max_length=40)
    name: str | None = Field(default=None, max_length=80)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    severity: Literal["critical", "high", "medium", "low"]
    description: str = Field(min_length=3, max_length=300)


class FaultNote(BaseModel):
    note: str | None = Field(default=None, max_length=200)


@app.get(f"{API}/faults/live", tags=["faults"])
def faults_live(offset_h: int = Query(0, ge=-72, le=71), include_predicted: bool = True, include_resolved: bool = False, p: Principal = R):
    return fh.live(svc(), offset_h, include_predicted, include_resolved)


@app.get(f"{API}/faults/assets", tags=["faults"])
def faults_assets(p: Principal = R):
    return fh.assets_view(svc())


@app.get(f"{API}/faults/locate", tags=["faults"])
def faults_locate(lat: float = Query(ge=-5.2, le=5.6), lon: float = Query(ge=33.5, le=42.5), p: Principal = R):
    return fh.locate(svc(), lat, lon)


@app.post(f"{API}/faults/report", tags=["faults"])
def faults_report(b: FaultReport, p: Principal = Depends(require("analyst"))):
    try:
        r = fh.add_report(svc(), p.key_id, b.asset_type, b.asset_id, b.lat, b.lon, b.severity, b.description, b.name)
    except fh.FaultError as e:
        raise HTTPException(422, detail={"code": "invalid_fault_report", "message": str(e)})
    audit().append("fault_reported", p.key_id, p.role, {"id": r["id"], "lat": r["lat"], "lon": r["lon"], "severity": b.severity})
    return r


@app.post(f"{API}/faults/reports/{{rid}}/resolve", tags=["faults"])
def faults_resolve(rid: str, p: Principal = Depends(require("analyst"))):
    if not fh.resolve_report(rid):
        raise HTTPException(404, detail={"code": "not_found", "message": "No open report with that id."})
    audit().append("fault_resolved", p.key_id, p.role, {"id": rid})
    return {"id": rid, "status": "resolved"}


@app.post(f"{API}/faults/ack", tags=["faults"])
def faults_ack(alert_id: str = Query(min_length=3, max_length=80), b: FaultNote | None = None, p: Principal = Depends(require("analyst"))):
    fh.acknowledge(alert_id, p.key_id, b.note if b else None)
    audit().append("fault_acknowledged", p.key_id, p.role, {"alert_id": alert_id})
    return {"alert_id": alert_id, "status": "acknowledged"}


@app.post(f"{API}/faults/dispatch", tags=["faults"])
def faults_dispatch(alert_id: str = Query(min_length=3, max_length=80), offset_h: int = Query(0, ge=-72, le=71), p: Principal = Depends(require("analyst"))):
    r = fh.dispatch_request(svc(), alert_id, p.key_id, offset_h)
    if r is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "That alert is not in the current feed."})
    audit().append("fault_dispatch_requested", p.key_id, p.role, {"alert_id": alert_id, "recommendation_id": r["recommendation_id"]})
    return {**r, "notice": "Sent to the approval queue. POWERIQ cannot dispatch crews or operate equipment."}


class CommunityConcern(BaseModel):
    kind: Literal["smoke", "fire", "sparking", "missing_component", "damaged", "pole_down", "vandalism", "sagging_line", "tree_on_line", "other"]
    lat: float = Field(ge=-5.2, le=5.6)
    lon: float = Field(ge=33.5, le=42.5)
    description: str = Field(default="", max_length=300)


class Detection(BaseModel):
    detection_id: str | None = Field(default=None, max_length=60)
    source: Literal["camera", "satellite", "drone", "other"]
    camera_id: str | None = Field(default=None, max_length=60)
    label: Literal["smoke", "fire", "sparking", "missing_component", "damaged", "pole_down", "vandalism", "sagging_line", "tree_on_line", "other"]
    object: str | None = Field(default=None, max_length=40, description="e.g. transformer, pole, conductor, insulator")
    confidence: float = Field(ge=0, le=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    observed_at: str | None = Field(default=None, max_length=30)
    image_ref: str | None = Field(default=None, max_length=200, description="Reference or URL of the frame/tile, not the image itself")
    model_version: str | None = Field(default=None, max_length=40)


class DetectionBatch(BaseModel):
    detections: list[Detection] = Field(min_length=1, max_length=100)


@app.post(f"{API}/faults/community", tags=["faults"])
def faults_community(b: CommunityConcern, p: Principal = R):
    try:
        r = fh.add_community(svc(), p.key_id, b.kind, b.lat, b.lon, b.description)
    except fh.FaultError as e:
        raise HTTPException(429 if "Too many" in str(e) else 422, detail={"code": "community_report_rejected", "message": str(e)})
    audit().append("community_concern", p.key_id, p.role, {"id": r["id"], "kind": b.kind, "lat": b.lat, "lon": b.lon})
    return r


@app.post(f"{API}/faults/community/{{cid}}/review", tags=["faults"])
def faults_community_review(cid: str, action: Literal["verify", "dismiss"], p: Principal = Depends(require("analyst"))):
    if not fh.set_community_status(cid, "verified" if action == "verify" else "dismissed"):
        raise HTTPException(404, detail={"code": "not_found", "message": "No open community report with that id."})
    audit().append("community_reviewed", p.key_id, p.role, {"id": cid, "action": action})
    return {"id": cid, "status": "verified" if action == "verify" else "dismissed"}


@app.post(f"{API}/faults/detections", tags=["faults"])
def faults_detections(b: DetectionBatch, p: Principal = Depends(require("analyst"))):
    r = fh.add_detections(svc(), p.key_id, [d.model_dump() for d in b.detections])
    audit().append("fault_detections_ingested", p.key_id, p.role, {"accepted": r["accepted"], "skipped": len(r["skipped"])})
    return r


@app.post(f"{API}/faults/detections/{{did}}/review", tags=["faults"])
def faults_detection_review(did: str, action: Literal["confirm", "dismiss"], p: Principal = Depends(require("analyst"))):
    if not fh.set_detection_status(did, "confirmed" if action == "confirm" else "dismissed"):
        raise HTTPException(404, detail={"code": "not_found", "message": "No open detection with that id."})
    audit().append("detection_reviewed", p.key_id, p.role, {"id": did, "action": action})
    return {"id": did, "status": "confirmed" if action == "confirm" else "dismissed"}


# ------------------------------------------------------------------ map
@app.get(f"{API}/map/cells", tags=["map"])
def map_cells(hour: int = Query(0, ge=0, le=71), p: Principal = R):
    return svc().map_cells(hour)


@app.get(f"{API}/map/assets", tags=["map"])
def map_assets(hour: int = Query(0, ge=0, le=71), p: Principal = R):
    return svc().map_assets(hour)


@app.get(f"{API}/map/timeline", tags=["map"])
def map_timeline(p: Principal = R):
    return svc().map_timeline()


# ------------------------------------------------------------------ storage and scenarios
class StorageRequest(BaseModel):
    power_mw: float = Field(gt=0, le=2000)
    energy_mwh: float = Field(gt=0, le=10000)
    round_trip_eff: float = Field(0.88, gt=0.5, le=1)
    reserve_frac: float = Field(0.15, ge=0, lt=0.9)
    horizon_hours: int = Field(24, ge=6, le=72)
    degradation_cost: float = Field(4.0, ge=0, le=100)


@app.post(f"{API}/storage/optimize", tags=["storage"])
def storage_optimize(req: StorageRequest, p: Principal = A):
    r = svc().storage_optimize(req.power_mw, req.energy_mwh, req.round_trip_eff, req.reserve_frac, req.horizon_hours, req.degradation_cost)
    audit().append("storage_optimize", p.key_id, p.role, req.model_dump() | {"status": r["status"]})
    if not r.get("feasible"):
        raise HTTPException(422, detail={"code": "infeasible", "message": f"No feasible schedule ({r['status']})."})
    return r


class StorageOption(BaseModel):
    power_mw: float = Field(gt=0, le=2000)
    energy_mwh: float = Field(gt=0, le=10000)


class StorageCompare(BaseModel):
    options: list[StorageOption] = Field(min_length=2, max_length=6)


@app.post(f"{API}/storage/compare", tags=["storage"])
def storage_compare(req: StorageCompare, p: Principal = A):
    r = svc().storage_compare([o.model_dump() for o in req.options])
    audit().append("storage_compare", p.key_id, p.role, {"n": len(req.options)})
    return r


class GenReduction(BaseModel):
    source: Literal["geothermal", "hydro", "wind", "solar", "thermal", "import"]
    pct: float = Field(gt=0, le=100)


class Outage(BaseModel):
    plant_id: str
    start_hour: int = Field(0, ge=0, le=71)
    duration_h: int = Field(24, ge=1, le=72)
    fraction: float = Field(1.0, gt=0, le=1)


class LineDerate(BaseModel):
    line_id: str
    derate_pct: float = Field(gt=0, le=100)


class BatteryIn(BaseModel):
    power_mw: float = Field(gt=0, le=2000)
    energy_mwh: float = Field(gt=0, le=10000)
    bus_id: str = "EMBAKASI"


class ScenarioRequest(BaseModel):
    name: str = Field("custom", max_length=80)
    horizon_hours: int = Field(24, ge=6, le=72)
    generation_reductions: list[GenReduction] = []
    plant_outages: list[Outage] = []
    line_derates: list[LineDerate] = []
    demand_increase_pct: float = Field(0.0, ge=-30, le=60)
    demand_region: str | None = None
    curtail_renewables_pct: float = Field(0.0, ge=0, le=100)
    battery: BatteryIn | None = None


def _register_recs(source: str, recs: list[str], actor: str, ctx: dict) -> list[dict]:
    out = []
    for r in recs:
        ex = [x for x in audit().list_recommendations("pending", 200) if x["text"] == r]
        out.append({"id": ex[0]["id"] if ex else audit().add_recommendation(source, r, actor, ctx), "text": r, "status": "pending"})
    return out


@app.get(f"{API}/scenarios/presets", tags=["scenarios"])
def scenario_presets(p: Principal = R):
    return {"presets": [
        {"name": "Geothermal output -15%", "generation_reductions": [{"source": "geothermal", "pct": 15}]},
        {"name": "Evening demand surge +10%", "demand_increase_pct": 10},
        {"name": "Dry spell: hydro -35%", "generation_reductions": [{"source": "hydro", "pct": 35}]},
        {"name": "Olkaria V outage", "plant_outages": [{"plant_id": "OLK5", "start_hour": 0, "duration_h": 24}]},
        {"name": "LTWP corridor derated 50%", "line_derates": [{"line_id": "LTWP_HUB-SUSWA", "derate_pct": 50}]},
        {"name": "Compound stress + 150 MW / 600 MWh battery", "generation_reductions": [{"source": "geothermal", "pct": 15}, {"source": "hydro", "pct": 30}],
         "demand_increase_pct": 8, "battery": {"power_mw": 150, "energy_mwh": 600}}],
        "plants": svc().plants[["plant_id", "name", "source"]].to_dict("records"),
        "lines": svc().net.lines.line_id.tolist(), "regions": [r.id for r in __import__("app.geo", fromlist=["REGIONS"]).REGIONS]}


@app.post(f"{API}/scenarios/run", tags=["scenarios"])
def scenario_run(req: ScenarioRequest, p: Principal = A):
    body = req.model_dump(exclude_none=True)
    r = svc().scenario(body)
    audit().append("scenario_run", p.key_id, p.role, {"name": req.name, "applied": r["applied_changes"], "unserved_delta_mwh": r["delta_vs_baseline"]["unserved_energy_mwh"]})
    recs = []
    if r["warnings"]:
        recs = _register_recs("scenario", [f"Scenario '{req.name}': {r['warnings'][0]} Review mitigation options with system operations."], p.key_id, {"scenario": req.name})
    r["recommendation_records"] = recs
    return r


# ------------------------------------------------------------------ rag
class RagQuery(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    category: str | None = None
    k: int = Field(4, ge=1, le=8)


class RagDoc(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{2,60}$")
    content: str = Field(min_length=20, max_length=200_000)


@app.post(f"{API}/rag/query", tags=["rag"])
def rag_query(req: RagQuery, p: Principal = R):
    r = get_kb().query(req.question, k=req.k, category=req.category)
    audit().append("rag_query", p.key_id, p.role, {"q": req.question, "hits": len(r["results"])})
    return r


@app.get(f"{API}/rag/documents", tags=["rag"])
def rag_docs(p: Principal = R):
    return {"documents": get_kb().documents()}


@app.post(f"{API}/rag/ingest", tags=["rag"])
def rag_ingest(doc: RagDoc, p: Principal = A):
    n = get_kb().add_document(doc.doc_id, doc.content)
    audit().append("rag_ingest", p.key_id, p.role, {"doc_id": doc.doc_id, "chunks": n})
    return {"doc_id": doc.doc_id, "chunks": n}


# ------------------------------------------------------------------ copilot and agents
class ChatRequest(BaseModel):
    message: str = Field(min_length=3, max_length=1000)
    conversation_id: str | None = Field(None, max_length=40)


@app.post(f"{API}/copilot/chat", tags=["copilot"])
def copilot_chat(req: ChatRequest, p: Principal = A):
    return copilot_mod.answer(req.message, svc(), agents(), audit(), p.key_id, p.role, req.conversation_id)


@app.get(f"{API}/agents", tags=["agents"])
def agent_status(p: Principal = R):
    return {"agents": agents().status(), "policy": "All tools are read-only. No agent can send commands to infrastructure."}


@app.post(f"{API}/agents/chief/assess", tags=["agents"])
def chief_assess(inp: AgentInput, p: Principal = A):
    out = agents().chief.assess(inp.question, inp.params.get("agents"), p.key_id, p.role, audit())
    out["recommendation_records"] = _register_recs("chief", out["recommendations"], p.key_id, {"question": inp.question})
    return out


@app.post(API + "/agents/{name}/run", tags=["agents"])
def agent_run(name: str, inp: AgentInput, p: Principal = A):
    if name not in agents().agents:
        raise HTTPException(404, detail={"code": "unknown_agent", "message": f"No agent named {name}."})
    out = agents().run(name, inp, p.key_id, p.role, audit())
    return out.model_dump()


# ------------------------------------------------------------------ audit and approvals
@app.get(f"{API}/audit/events", tags=["audit"])
def audit_events(limit: int = Query(100, ge=1, le=500), event_type: str | None = None, p: Principal = A):
    return {"events": audit().events(limit, event_type)}


@app.get(f"{API}/audit/verify", tags=["audit"])
def audit_verify(p: Principal = A):
    return audit().verify()


@app.get(f"{API}/recommendations", tags=["audit"])
def recommendations(status: Literal["pending", "approved", "rejected"] | None = None, p: Principal = R):
    return {"recommendations": audit().list_recommendations(status)}


class Decision(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str | None = Field(None, max_length=500)


@app.post(API + "/recommendations/{rid}/decision", tags=["audit"])
def decide(rid: str, d: Decision, p: Principal = AP):
    r = audit().decide(rid, d.decision, p.key_id, d.note)
    if r is None:
        raise HTTPException(404, detail={"code": "not_found_or_decided", "message": "Recommendation not found or already decided."})
    audit().append("recommendation_decision", p.key_id, p.role, {"id": rid, "decision": d.decision, "note": d.note})
    return r | {"notice": "Approval is a recorded decision only. POWERIQ cannot execute operational actions."}


# ------------------------------------------------------------------ telemetry validation
class TelemetryPoint(BaseModel):
    asset_id: str = Field(pattern=r"^[A-Z]{3}-\d{2}$")
    time: datetime
    load_pct: float
    vibration: float
    temperature: float
    pressure: float
    flow: float

    @field_validator("load_pct")
    @classmethod
    def _load(cls, v):
        if not (0 <= v <= 130):
            raise ValueError("load_pct must be within 0..130")
        return v

    @field_validator("vibration", "temperature", "pressure", "flow")
    @classmethod
    def _finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class TelemetryBatch(BaseModel):
    points: list[TelemetryPoint] = Field(min_length=1, max_length=500)


@app.post(f"{API}/telemetry/validate", tags=["anomaly"])
def telemetry_validate(batch: TelemetryBatch, p: Principal = A):
    ab = svc().ab
    rows, rejected = [], []
    for i, pt in enumerate(batch.points):
        if pt.asset_id not in ab.stats:
            rejected.append({"index": i, "reason": f"unknown asset {pt.asset_id}"})
            continue
        if not (0 <= pt.vibration < 50 and -40 <= pt.temperature < 400 and 0 <= pt.pressure < 200 and 0 <= pt.flow < 5000):
            rejected.append({"index": i, "reason": "value outside physical plausibility range"})
            continue
        rows.append(pt)
    scored = []
    if rows:
        df = pd.DataFrame([{"asset_id": r.asset_id, "load_pct": r.load_pct, "vibration": r.vibration, "temperature": r.temperature,
                            "pressure": r.pressure, "flow": r.flow} for r in rows])
        X, _ = ml._anomaly_features(df, ab.stats)
        s = -ab.model.score_samples(X)
        order = pd.concat([g for _, g in df.groupby("asset_id", sort=False)])
        idx_map = {i: r for i, r in enumerate(rows)}
        sorted_rows = [rows[i] for i in order.index]
        scored = [{"asset_id": r.asset_id, "time": str(r.time), "anomaly_score": round(float(v), 3), "flagged": bool(v >= ab.threshold)}
                  for r, v in zip(sorted_rows, s)]
    audit().append("telemetry_validate", p.key_id, p.role, {"accepted": len(rows), "rejected": len(rejected)})
    return {"accepted": len(rows), "rejected": rejected, "scored": scored, "threshold": ab.threshold}


# ------------------------------------------------------------------ minimal MCP (JSON-RPC) endpoint
class RpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = {}


@app.post(f"{API}/mcp", tags=["mcp"])
def mcp(req: RpcRequest, p: Principal = A):
    tools = agents().tools
    def ok(result):
        return {"jsonrpc": "2.0", "id": req.id, "result": result}
    if req.method == "initialize":
        return ok({"protocolVersion": "2024-11-05", "serverInfo": {"name": "poweriq-mcp", "version": "1.0.0"}, "capabilities": {"tools": {}}})
    if req.method == "tools/list":
        return ok({"tools": [{"name": t.name, "description": t.description + " (read-only)", "inputSchema": t.schema} for t in tools.values()]})
    if req.method == "tools/call":
        name, args = req.params.get("name"), req.params.get("arguments", {})
        if name not in tools:
            return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32602, "message": f"unknown tool {name}"}}
        try:
            res = tools[name].fn(**args)
        except Exception as e:
            return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}}
        audit().append("mcp_tool_call", p.key_id, p.role, {"tool": name})
        import json
        return ok({"content": [{"type": "text", "text": json.dumps(res, default=str)[:60000]}], "isError": False})
    return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32601, "message": "method not found"}}


# ------------------------------------------------------------------ static frontend (single-container deploy)
_dist = Path(os.getenv("POWERIQ_FRONTEND_DIST", Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"))
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        f = _dist / full_path
        return FileResponse(f if f.is_file() else _dist / "index.html")
