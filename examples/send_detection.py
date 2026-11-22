"""Example: send detections from an externally trained model (cameras, satellite tiles, drones) to POWERIQ.
Run:  POWERIQ_URL=http://localhost:8000 POWERIQ_KEY=demo-analyst-key python examples/send_detection.py
Your model code calls post_detections([...]) with whatever it found. POWERIQ does not run or train the model."""
import json, os, urllib.request

URL = os.environ.get("POWERIQ_URL", "http://localhost:8000")
KEY = os.environ.get("POWERIQ_KEY", "demo-analyst-key")


def post_detections(detections: list[dict]) -> dict:
    req = urllib.request.Request(f"{URL}/api/v1/faults/detections", data=json.dumps({"detections": detections}).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": KEY}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


if __name__ == "__main__":
    print(json.dumps(post_detections([{
        "source": "camera", "camera_id": "cam-nairobi-east-03", "label": "smoke", "object": "transformer",
        "confidence": 0.93, "lat": -1.2921, "lon": 36.8219, "observed_at": "2026-11-21T06:12:00+03:00",
        "image_ref": "frames/cam-nairobi-east-03/20261121T061200.jpg", "model_version": "smoke-v1"}]), indent=2))
