import json

import pytest
from conftest import VIEWER

from app import query_studio as qs


def ask(client, q):
    r = client.post("/api/v1/query/ask", json={"question": q}, headers=VIEWER)
    assert r.status_code == 200, r.text[:300]
    return r.json()


def test_catalog(client):
    d = client.get("/api/v1/query/catalog", headers=VIEWER).json()
    names = {x["name"] for x in d["datasets"]}
    assert {"plants", "forecast_system", "feeder_losses", "units_risk", "demand_hourly"} <= names
    assert client.get("/api/v1/query/catalog").status_code == 401


@pytest.mark.parametrize("q,ds", [("total installed capacity by source", "plants"), ("non-technical loss by region", "feeder_losses"),
                                  ("top 5 units by failure risk", "units_risk"), ("curtailed energy by plant", "curtailment_events"),
                                  ("average demand by hour of day", "demand_hourly"), ("max line loading by line", "line_loading_forecast"),
                                  ("what is the lowest reserve margin in the next 72 hours", "forecast_system")])
def test_local_planner(client, q, ds):
    d = ask(client, q)
    assert d["mode"] == "local-planner" and not d["errors"], d
    assert d["results"][0]["dataset"] == ds and d["results"][0]["row_count"] > 0
    assert d["narrative"]


def test_local_values_correct(client, svc):
    d = ask(client, "total capacity by source")
    r = d["results"][0]
    got = {row[0]: row[1] for row in r["rows"]}
    exp = svc.plants.groupby("source").capacity_mw.sum()
    for k, v in exp.items():
        assert abs(got[k] - v) < 0.01


def test_run_validation(client):
    bad = {"dataset": "plants", "metrics": [{"col": "nope", "agg": "sum"}]}
    assert client.post("/api/v1/query/run", json=bad, headers=VIEWER).status_code == 422
    assert client.post("/api/v1/query/run", json={"dataset": "nope"}, headers=VIEWER).status_code == 422
    assert client.post("/api/v1/query/run", json={"dataset": "plants", "metrics": [{"col": "source", "agg": "sum"}]}, headers=VIEWER).status_code == 422
    assert client.post("/api/v1/query/run", json={"dataset": "plants", "limit": 5000}, headers=VIEWER).status_code == 422
    assert client.post("/api/v1/query/run", json={"dataset": "plants", "filters": [{"col": "source", "op": "DROP", "value": "x"}]}, headers=VIEWER).status_code == 422


def test_run_pivot_and_filter(client):
    spec = {"dataset": "demand_hourly", "time_bucket": "day", "group_by": ["region"], "filters": [{"col": "period", "op": "=", "value": "history"}],
            "metrics": [{"col": "demand_mw", "agg": "mean", "alias": "avg"}], "chart": "line", "limit": 200}
    r = client.post("/api/v1/query/run", json=spec, headers=VIEWER).json()
    assert r["chart"]["type"] == "line" and len(r["chart"]["series"]) >= 2 and r["chart"]["x"] == "time_bucket"


def test_dashboard_default(client):
    d = client.post("/api/v1/query/dashboard", json={}, headers=VIEWER).json()
    assert len(d["tiles"]) == 6 and not d["errors"] and d["mode"] == "local-default"


def test_openai_path_mocked(client, monkeypatch):
    plan = {"queries": [{"dataset": "plants", "title": "Capacity", "group_by": ["source"], "metrics": [{"col": "capacity_mw", "agg": "sum", "alias": "mw"}], "chart": "bar"}], "intent": "capacity"}
    calls = []

    def fake(messages, json_mode):
        calls.append(json_mode)
        return (json.dumps(plan) if json_mode else "Geothermal leads. Synthetic data."), None
    monkeypatch.setattr(qs, "_chat", fake)
    d = ask(client, "which technology has the most capacity?")
    assert d["mode"] == "openai" and d["narrative"].startswith("Geothermal") and calls == [True, False]


def test_openai_bad_plan_falls_back(client, monkeypatch):
    monkeypatch.setattr(qs, "_chat", lambda m, j: ('{"queries":[{"dataset":"plants","metrics":[{"col":"x","agg":"eval"}]}]}', None))
    d = ask(client, "total capacity by source")
    assert d["mode"] == "local-planner" and "invalid spec" in d["llm_status"]


def test_openai_hallucinated_column_reported(client, monkeypatch):
    plan = {"queries": [{"dataset": "plants", "metrics": [{"col": "ghost", "agg": "sum"}]}], "intent": "x"}
    monkeypatch.setattr(qs, "_chat", lambda m, j: (json.dumps(plan), None) if j else ("n", None))
    d = ask(client, "anything")
    assert d["mode"] == "openai" and d["errors"] and not d["results"]


def test_out_of_scope_refused_locally(client):
    for q in ["Who won the football match yesterday?", "Write me a poem about the ocean", "What is the capital of France?"]:
        d = ask(client, q)
        assert d["kind"] == "out_of_scope" and d["results"] == [] and "outside the scope" in d["narrative"], q


def test_capabilities_question(client):
    d = ask(client, "What can you do?")
    assert d["kind"] == "capabilities" and "plants" in d["narrative"]


def test_analysis_mode_local(client):
    d = ask(client, "Why could the reserve margin fall in the next 72 hours and what should we do?")
    assert d["kind"] == "analysis" and d["findings"] and d["narrative"]


def test_multi_part_question(client):
    d = ask(client, "Compare curtailment by plant and non-technical loss by region")
    assert len(d["results"]) == 2 and {r["dataset"] for r in d["results"]} == {"curtailment_events", "feeder_losses"}


def test_analysis_statistics(client):
    d = ask(client, "Average demand by hour of day")
    assert d["analysis"] and d["analysis"][0]["stats"]
    assert any("peak" in n or "rising" in n or "falling" in n for n in d["analysis"][0]["notes"])


def test_status_and_test_endpoints(client):
    st = client.get("/api/v1/query/status", headers=VIEWER).json()
    assert st["openai_configured"] is False and st["tested_against_real_api"] is False
    assert client.post("/api/v1/query/test-openai", headers=VIEWER).status_code == 403
    r = client.post("/api/v1/query/test-openai", headers={"X-API-Key": "demo-analyst-key"}).json()
    assert r["ok"] is False and "OPENAI_API_KEY" in r["error"]


def test_openai_scope_out_and_analysis_mocked(client, monkeypatch):
    monkeypatch.setattr(qs, "_chat", lambda m, j: ('{"scope":"out","mode":"query","queries":[]}', None))
    d = ask(client, "tell me a joke about cats")
    assert d["mode"] == "openai" and d["kind"] == "out_of_scope"
    plan = {"scope": "in", "mode": "analysis", "queries": [{"dataset": "forecast_system", "time_bucket": "hour", "metrics": [{"col": "reserve_margin_pct", "agg": "min"}], "chart": "line"}], "intent": "reserve"}
    monkeypatch.setattr(qs, "_chat", lambda m, j: (json.dumps(plan), None) if j else ("Reserve is adequate; synthetic data.", None))
    d = ask(client, "analyse reserve risk")
    assert d["mode"] == "openai" and d["kind"] == "analysis" and d["findings"] and d["narrative"].startswith("Reserve")
