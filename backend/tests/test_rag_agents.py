import pytest
from app.agents import AgentInput, AgentRegistry
from app.rag import KnowledgeBase


def test_retrieval_with_citations_and_filter():
    kb = KnowledgeBase()
    r = kb.query("how are non-technical losses estimated", k=3)
    assert r["answerable"] and r["results"][0]["doc_id"] == "loss_classification"
    assert r["results"][0]["citation"].startswith("[loss_classification#")
    only = kb.query("vibration bearing", category="maintenance")
    assert all(x["category"] == "maintenance" for x in only["results"])


def test_unsupported_question_is_not_answered():
    r = KnowledgeBase().query("what is the EPRA tariff for 2025 licensing requirement")
    assert not r["answerable"] and r["results"] == []


def test_document_ingestion():
    kb = KnowledgeBase()
    n = kb.add_document("manual_x", "---\ntitle: Pump X manual\ncategory: maintenance\nsource: test\n---\nThe Pump X impeller clearance must be inspected every 4000 hours using a feeler gauge.")
    assert n >= 1
    r = kb.query("pump impeller clearance inspection", k=1)
    assert r["answerable"] and r["results"][0]["doc_id"] == "manual_x"


def test_all_specialists_run_with_typed_output(svc):
    reg = AgentRegistry(svc)
    for name in reg.agents:
        out = reg.run(name, AgentInput(question="status"))
        assert out.status == "ok", (name, out.error)
        assert out.findings and out.confidence_definition and out.timestamp
        assert {f.kind for f in out.findings} <= {"measured", "prediction", "assumption", "recommendation"}
        assert set(out.tools_used) <= set(reg.agents[name].tools)


def test_agent_cannot_call_unlisted_tool(svc):
    reg = AgentRegistry(svc)
    with pytest.raises(PermissionError):
        reg.agents["demand_forecasting"].call("run_scenario", [])


def test_no_agent_tool_can_actuate(svc):
    reg = AgentRegistry(svc)
    assert all(t.read_only for t in reg.tools.values())
    assert not any(w in t.name for t in reg.tools.values() for w in ("command", "dispatch", "trip", "set_", "write"))


def test_agent_error_is_contained(svc, monkeypatch):
    reg = AgentRegistry(svc)
    monkeypatch.setattr(reg.agents["loss_detection" if False else "electricity_loss_detection"], "run", lambda inp: 1 / 0)
    out = reg.run("electricity_loss_detection", AgentInput())
    assert out.status == "error" and "ZeroDivisionError" in out.error


def test_agent_timeout_is_contained(svc, monkeypatch):
    import time
    from app import security
    reg = AgentRegistry(svc)
    monkeypatch.setattr(security.get_settings(), "agent_timeout_s", 0.2)
    monkeypatch.setattr(reg.agents["climate_intelligence"], "run", lambda inp: time.sleep(1.5))
    out = reg.run("climate_intelligence", AgentInput())
    assert out.status == "error" and out.error == "timeout"


def test_chief_coordinates_and_reports_missing_info(svc):
    reg = AgentRegistry(svc)
    out = reg.chief.assess("overall", None, "t", "viewer", None)
    assert set(out["agents_run"]) == set(reg.agents)
    assert out["overall_risk_level"] in ("low", "moderate", "elevated", "high")
    assert any("SCADA" in m for m in out["missing_information"])
    assert out["coupled_scenario"] is not None and out["approval_required"]


def test_chief_degrades_when_a_specialist_fails(svc, monkeypatch):
    reg = AgentRegistry(svc)
    monkeypatch.setattr(reg.agents["grid_intelligence"], "run", lambda inp: 1 / 0)
    out = reg.chief.assess("overall", None, "t", "viewer", None)
    assert any("grid_intelligence" in m for m in out["missing_information"])
