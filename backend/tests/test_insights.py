from conftest import ANALYST, VIEWER

EP = ["scorecard", "reserve-risk", "curtailment-storage", "climate-stress", "theft-planner", "plant-fleet", "demand-structure", "analysis"]


def test_all_insight_endpoints(client):
    for e in EP:
        r = client.get(f"/api/v1/insights/{e}", headers=VIEWER)
        assert r.status_code == 200, (e, r.text[:200])
    assert client.get("/api/v1/insights/scorecard").status_code == 401


def test_reserve_risk_sane(client):
    d = client.get("/api/v1/insights/reserve-risk", headers=VIEWER).json()
    assert 0 <= d["p_any_shortfall_72h"] <= 1 and len(d["hourly"]) == 72
    assert all(0 <= h["lolp"] <= 1 for h in d["hourly"])
    assert d["p95_unserved_mwh_72h"] >= 0 and d["expected_unserved_mwh_72h"] >= 0


def test_storage_options_reduce_curtailment(client):
    d = client.get("/api/v1/insights/curtailment-storage", headers=VIEWER).json()
    for o in d["battery_options"]:
        assert o["curtailment_recovered_mwh"] >= -1e-6


def test_drought_monotonic(client):
    sc = client.get("/api/v1/insights/climate-stress", headers=VIEWER).json()["scenarios"]
    th = [x["thermal_mwh"] + x["unserved_mwh"] for x in sc[:4]]
    assert th == sorted(th)


def test_fleet_and_demand(client):
    f = client.get("/api/v1/insights/plant-fleet", headers=VIEWER).json()
    assert len(f["plants"]) == 27 and sum(c["n_plants"] for c in f["clusters"]) == 27
    d = client.get("/api/v1/insights/demand-structure", headers=VIEWER).json()
    assert len(d["hourly_profile_mw"]) == 24 and 0 <= d["seasonal_strength"] <= 1


def test_analysis_and_propose_rbac(client):
    a = client.get("/api/v1/insights/analysis", headers=VIEWER).json()
    assert a["mode"] == "local-template" and a["findings"]
    fid = a["findings"][0]["id"]
    assert client.post("/api/v1/insights/propose", json={"finding_id": fid}, headers=VIEWER).status_code == 403
    r = client.post("/api/v1/insights/propose", json={"finding_id": fid}, headers=ANALYST)
    assert r.status_code == 200 and r.json()["status"] == "pending"
    assert client.post("/api/v1/insights/propose", json={"finding_id": "F-NOPE"}, headers=ANALYST).status_code == 404
