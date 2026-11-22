from app import fault_hub as fh
from app.service import get_service
from conftest import ANALYST, VIEWER

API = "/api/v1"


def test_live_alerts_have_coordinates_and_order():
    r = fh.live(get_service())
    xs = r["alerts"]
    assert xs and all(-5.2 <= a["lat"] <= 5.6 and 33.5 <= a["lon"] <= 42.5 for a in xs)
    assert [a["priority"] for a in xs] == sorted((a["priority"] for a in xs), reverse=True)
    assert all(a["asset_type"] in ("plant", "substation", "line", "other") for a in xs)
    assert "Simulated" in r["notice"]


def test_locate_nairobi():
    assert fh.locate(get_service(), -1.29, 36.82)["region"] == "NAIROBI"


def test_report_validation_and_flow(client):
    bad = client.post(f"{API}/faults/report", headers=ANALYST, json={"asset_type": "other", "lat": 40, "lon": 0, "severity": "high", "description": "outside"})
    assert bad.status_code == 422
    ok = client.post(f"{API}/faults/report", headers=ANALYST, json={"asset_type": "other", "lat": -1.0, "lon": 36.9, "severity": "critical", "description": "Pole down, line sparking"})
    assert ok.status_code == 200, ok.text
    rid = ok.json()["id"]
    live = client.get(f"{API}/faults/live", headers=VIEWER).json()
    a = next(x for x in live["alerts"] if x["id"] == "rep:" + rid)
    assert a["status"] == "reported" and a["severity"] == "critical"
    assert client.post(f"{API}/faults/ack", params={"alert_id": a["id"]}, headers=ANALYST, json={"note": "crew en route"}).status_code == 200
    d = client.post(f"{API}/faults/dispatch", params={"alert_id": a["id"]}, headers=ANALYST)
    assert d.status_code == 200 and d.json()["status"] == "pending"
    assert client.post(f"{API}/faults/reports/{rid}/resolve", headers=ANALYST).status_code == 200
    assert client.post(f"{API}/faults/reports/{rid}/resolve", headers=ANALYST).status_code == 404


def test_asset_report_and_permissions(client):
    r = client.post(f"{API}/faults/report", headers=ANALYST, json={"asset_type": "line", "asset_id": "SUSWA-EMBAKASI", "severity": "high", "description": "Conductor sag"})
    assert r.status_code == 200
    assert client.post(f"{API}/faults/report", headers=VIEWER, json={"asset_type": "other", "lat": -1, "lon": 36.9, "severity": "low", "description": "xyz"}).status_code == 403
    assert client.post(f"{API}/faults/report", headers=ANALYST, json={"asset_type": "plant", "asset_id": "NOPE", "severity": "low", "description": "abc"}).status_code == 422


def test_model_detection_to_headline_and_dispatch(client):
    body = {"detections": [
        {"detection_id": "T-HIGH", "source": "camera", "camera_id": "cam-nbi-east-3", "label": "smoke", "object": "transformer", "confidence": 0.93, "lat": -1.2921, "lon": 36.8219, "model_version": "x1"},
        {"detection_id": "T-MID", "source": "satellite", "label": "missing_component", "object": "transformer", "confidence": 0.62, "lat": -0.5, "lon": 36.5},
        {"detection_id": "T-LOW", "source": "camera", "label": "smoke", "confidence": 0.2, "lat": -1.3, "lon": 36.8},
        {"detection_id": "T-BAD", "source": "camera", "label": "smoke", "confidence": 0.9, "lat": 10, "lon": 36.8}]}
    r = client.post(f"{API}/faults/detections", headers=ANALYST, json=body).json()
    assert r["accepted"] == 3 and len(r["skipped"]) == 1
    al = {x["id"]: x for x in client.get(f"{API}/faults/live", headers=VIEWER).json()["alerts"]}
    hi = al["det:T-HIGH"]
    assert hi["status"] == "detected" and hi["severity"] == "critical" and "Transformer smoking at" in hi["name"]
    assert "Dispatch a crew to the coordinates (-1.2921, 36.8219)" in hi["recommended_action"]
    assert al["det:T-MID"]["status"] == "unverified" and "det:T-LOW" not in al
    assert hi["priority"] > al["det:T-MID"]["priority"]
    assert client.post(f"{API}/faults/detections/T-MID/review", params={"action": "dismiss"}, headers=ANALYST).status_code == 200
    d = client.post(f"{API}/faults/dispatch", params={"alert_id": "det:T-HIGH"}, headers=ANALYST).json()
    assert "Transformer smoking" in d["text"] and d["status"] == "pending"


def test_community_flow_and_corroboration(client):
    c = {"kind": "smoke", "lat": -1.30, "lon": 36.90, "description": "Smoke from pole-mounted transformer"}
    r1 = client.post(f"{API}/faults/community", headers=VIEWER, json=c)
    assert r1.status_code == 200 and r1.json()["status"] == "unverified"
    a1 = next(x for x in client.get(f"{API}/faults/live", headers=VIEWER).json()["alerts"] if x["id"] == r1.json()["alert_id"])
    assert a1["status"] == "unverified" and "Verify first" in a1["recommended_action"]
    client.post(f"{API}/faults/community", headers=VIEWER, json=c)
    live = {x["id"]: x for x in client.get(f"{API}/faults/live", headers=VIEWER).json()["alerts"]}
    assert live[r1.json()["alert_id"]]["corroborating_reports"] >= 1 and live[r1.json()["alert_id"]]["severity"] == "critical"
    assert client.post(f"{API}/faults/community", headers=VIEWER, json={**c, "lat": 40}).status_code == 422
    assert client.post(f"{API}/faults/community/{r1.json()['id']}/review", params={"action": "verify"}, headers=VIEWER).status_code == 403
    assert client.post(f"{API}/faults/community/{r1.json()['id']}/review", params={"action": "verify"}, headers=ANALYST).json()["status"] == "verified"
    a2 = next(x for x in client.get(f"{API}/faults/live", headers=VIEWER).json()["alerts"] if x["id"] == r1.json()["alert_id"])
    assert a2["status"] == "reported"
