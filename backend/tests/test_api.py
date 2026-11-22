import sqlite3
import pytest
from app import copilot, security
from conftest import ANALYST, APPROVER, VIEWER

P = "/api/v1"


def test_health_open_but_api_protected(client):
    assert client.get("/health").status_code == 200
    r = client.get(f"{P}/overview")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated" and "request_id" in r.json()["error"]


def test_invalid_credentials(client):
    r = client.get(f"{P}/overview", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_credentials"


def test_bearer_token_accepted(client):
    assert client.get(f"{P}/overview", headers={"Authorization": "Bearer demo-viewer-key"}).status_code == 200


def test_rbac(client):
    assert client.post(f"{P}/scenarios/run", json={}, headers=VIEWER).status_code == 403
    assert client.get(f"{P}/audit/events", headers=VIEWER).status_code == 403
    assert client.post(f"{P}/copilot/chat", json={"message": "demand"}, headers=VIEWER).status_code == 403
    assert client.get(f"{P}/overview", headers=VIEWER).status_code == 200


@pytest.mark.parametrize("path", ["overview", "generation", "demand", "maintenance", "anomalies", "losses", "curtailment",
                                  "grid", "grid/n1", "climate", "usage", "metrics", "map/cells", "map/assets", "map/timeline",
                                  "agents", "scenarios/presets", "integrations/status", "forecast?horizon_hours=24", "rag/documents"])
def test_read_endpoints(client, path):
    r = client.get(f"{P}/{path}", headers=VIEWER)
    assert r.status_code == 200, r.text[:200]


def test_validation_errors_are_structured(client):
    r = client.post(f"{P}/storage/optimize", json={"power_mw": -5, "energy_mwh": 100}, headers=ANALYST)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert client.get(f"{P}/forecast?horizon_hours=5", headers=VIEWER).status_code == 422
    assert client.get(f"{P}/map/cells?hour=999", headers=VIEWER).status_code == 422
    r = client.post(f"{P}/scenarios/run", json={"plant_outages": [{"plant_id": "NOPE"}]}, headers=ANALYST)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_scenario_and_storage_endpoints(client):
    r = client.post(f"{P}/scenarios/run", json={"name": "geo", "generation_reductions": [{"source": "geothermal", "pct": 15}]}, headers=ANALYST)
    assert r.status_code == 200 and r.json()["approval_required"] is True
    r = client.post(f"{P}/storage/optimize", json={"power_mw": 100, "energy_mwh": 400}, headers=ANALYST)
    assert r.status_code == 200 and r.json()["schedule_valid"]
    r = client.post(f"{P}/storage/compare", json={"options": [{"power_mw": 50, "energy_mwh": 200}, {"power_mw": 150, "energy_mwh": 600}]}, headers=ANALYST)
    assert r.status_code == 200 and len(r.json()["options"]) == 2


def test_copilot_sections_and_provenance(client):
    r = client.post(f"{P}/copilot/chat", json={"message": "What is the predicted electricity demand in Kenya over the next 24 hours?"}, headers=ANALYST).json()
    for k in ("measured_facts", "model_predictions", "assumptions", "recommendations", "citations", "timestamp", "model_version", "uncertainty", "limitations"):
        assert k in r
    assert r["model_predictions"] and r["mode"] == "local-template"


def test_copilot_scenario_question(client):
    r = client.post(f"{P}/copilot/chat", json={"message": "What happens if geothermal output decreases by 15%?"}, headers=ANALYST).json()
    assert r["intent"] == "scenario" and r["scenario"]["applied_changes"] == ["geothermal output -15%"]


def test_copilot_refuses_without_evidence(client):
    r = client.post(f"{P}/copilot/chat", json={"message": "What is the EPRA tariff for 2025?"}, headers=ANALYST).json()
    assert r["intent"] == "no_evidence" and "don't have evidence" in r["answer"] and not r["citations"]


@pytest.mark.parametrize("q,intent", [("Which generation assets have the highest maintenance risk?", "maintenance"),
                                      ("Which renewable resources are being curtailed?", "curtail"),
                                      ("Which battery schedule minimizes peak demand?", "storage"),
                                      ("Compare alternative energy storage scenarios", "storage"),
                                      ("Which distribution regions show unusual energy losses?", "loss"),
                                      ("Generate a maintenance priority report", "report"),
                                      ("What is causing the simulated generation-demand gap?", "gap")])
def test_copilot_example_questions(client, q, intent):
    r = client.post(f"{P}/copilot/chat", json={"message": q}, headers=ANALYST)
    assert r.status_code == 200 and r.json()["intent"] == intent and r.json()["answer"]


def test_openai_failure_falls_back_to_local(client, monkeypatch):
    import httpx
    monkeypatch.setattr(security.get_settings(), "openai_api_key", "sk-test")
    def boom(*a, **k):
        raise httpx.ConnectTimeout("simulated outage")
    monkeypatch.setattr(copilot.httpx, "post", boom)
    r = client.post(f"{P}/copilot/chat", json={"message": "Which units have the highest maintenance risk?"}, headers=ANALYST).json()
    assert r["mode"] == "local-template" and "ConnectTimeout" in r["llm_status"]


def test_openai_success_path_uses_llm_text(client, monkeypatch):
    monkeypatch.setattr(security.get_settings(), "openai_api_key", "sk-test")
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "LLM NARRATIVE"}}]}
    monkeypatch.setattr(copilot.httpx, "post", lambda *a, **k: Resp())
    r = client.post(f"{P}/copilot/chat", json={"message": "Which units have the highest maintenance risk?"}, headers=ANALYST).json()
    assert r["mode"] == "openai" and r["answer"] == "LLM NARRATIVE" and r["model_predictions"]


def test_recommendation_approval_flow(client):
    client.post(f"{P}/agents/chief/assess", json={"question": "status"}, headers=ANALYST)
    recs = client.get(f"{P}/recommendations?status=pending", headers=VIEWER).json()["recommendations"]
    assert recs
    rid = recs[0]["id"]
    assert client.post(f"{P}/recommendations/{rid}/decision", json={"decision": "approved"}, headers=ANALYST).status_code == 403
    r = client.post(f"{P}/recommendations/{rid}/decision", json={"decision": "approved", "note": "ok"}, headers=APPROVER)
    assert r.status_code == 200 and r.json()["status"] == "approved" and "cannot execute" in r.json()["notice"]
    assert client.post(f"{P}/recommendations/{rid}/decision", json={"decision": "rejected"}, headers=APPROVER).status_code == 404


def test_audit_chain_detects_tampering(client):
    assert client.get(f"{P}/audit/verify", headers=ANALYST).json()["valid"] is True
    db = security.get_settings().audit_db
    con = sqlite3.connect(db)
    seq = con.execute("SELECT seq FROM audit ORDER BY seq LIMIT 1 OFFSET 2").fetchone()[0]
    original = con.execute("SELECT payload FROM audit WHERE seq=?", (seq,)).fetchone()[0]
    con.execute("UPDATE audit SET payload='{\"tampered\": true}' WHERE seq=?", (seq,))
    con.commit()
    v = client.get(f"{P}/audit/verify", headers=ANALYST).json()
    assert v["valid"] is False and v["broken_at_seq"] == seq
    con.execute("UPDATE audit SET payload=? WHERE seq=?", (original, seq))
    con.commit()
    con.close()
    assert client.get(f"{P}/audit/verify", headers=ANALYST).json()["valid"] is True


def test_auth_failures_are_audited(client):
    client.get(f"{P}/overview", headers={"X-API-Key": "bad-key-xyz"})
    ev = client.get(f"{P}/audit/events?event_type=auth_failure", headers=ANALYST).json()["events"]
    assert ev and "bad-key-xyz" not in str(ev)   # raw keys are never stored


def test_rate_limiter():
    lim = security.RateLimiter(3)
    assert [lim.check("k") for _ in range(3)] == [None, None, None]
    assert isinstance(lim.check("k"), int)
    assert lim.check("other") is None


def test_production_requires_keys(monkeypatch):
    monkeypatch.setenv("POWERIQ_ENV", "production")
    monkeypatch.delenv("POWERIQ_API_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        security.load_settings()
    monkeypatch.setenv("POWERIQ_API_KEYS", "k1:viewer,k2:bogus")
    with pytest.raises(RuntimeError):
        security.load_settings()


def test_telemetry_validation(client, svc):
    good = {"asset_id": "TUR-01", "time": "2026-11-18T10:00:00", "load_pct": 70, "vibration": 2.3, "temperature": 86, "pressure": 14, "flow": 80}
    bad_range = {**good, "vibration": 900}
    bad_asset = {**good, "asset_id": "TUR-99"}
    anomalous = {**good, "vibration": 6.5, "temperature": 120}
    r = client.post(f"{P}/telemetry/validate", json={"points": [good, bad_range, bad_asset, anomalous]}, headers=ANALYST).json()
    assert r["accepted"] == 2 and len(r["rejected"]) == 2
    assert r["scored"][1]["flagged"] is True
    assert client.post(f"{P}/telemetry/validate", json={"points": [{**good, "load_pct": 500}]}, headers=ANALYST).status_code == 422
    assert client.post(f"{P}/telemetry/validate", json={"points": []}, headers=ANALYST).status_code == 422


def test_rag_endpoints_and_ingest(client):
    r = client.post(f"{P}/rag/query", json={"question": "what does the DC approximation ignore"}, headers=VIEWER).json()
    assert r["answerable"] and r["results"][0]["citation"]
    assert client.post(f"{P}/rag/ingest", json={"doc_id": "manual_t", "content": "Transformer T1 buchholz relay alarm procedure: isolate and inspect before re-energising."}, headers=VIEWER).status_code == 403
    assert client.post(f"{P}/rag/ingest", json={"doc_id": "manual_t", "content": "Transformer T1 buchholz relay alarm procedure: isolate and inspect before re-energising."}, headers=ANALYST).status_code == 200


def test_mcp_endpoint(client):
    tools = client.post(f"{P}/mcp", json={"method": "tools/list", "id": 1}, headers=ANALYST).json()["result"]["tools"]
    assert len(tools) >= 10 and all("read-only" in t["description"] for t in tools)
    r = client.post(f"{P}/mcp", json={"method": "tools/call", "id": 2, "params": {"name": "launch_missiles", "arguments": {}}}, headers=ANALYST).json()
    assert r["error"]["code"] == -32602
    assert client.post(f"{P}/mcp", json={"method": "tools/list", "id": 1}, headers=VIEWER).status_code == 403


def test_integrations_are_honest(client):
    j = client.get(f"{P}/integrations/status", headers=VIEWER).json()["integrations"]
    status = {i["name"]: i["status"] for i in j}
    assert status["Microsoft Foundry IQ"] == "not_configured" and status["Live grid / SCADA"] == "never_connected"
