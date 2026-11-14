"""End-to-end check against a running backend: register a tiny dataset, train, poll, infer, zero-shot.

Run INSIDE the backend container (so the dataset path is visible to the trainer):
  docker exec zalu-backend python -m tests.e2e_backend   (repo copied to /app)
"""
import os, sys, time, urllib.request
import requests
from ultralytics.data.utils import check_det_dataset

API = os.environ.get("ZALU_API", "http://127.0.0.1:9100")
H = {"Authorization": f"Bearer {os.environ['ZALU_TOKEN']}"} if os.environ.get("ZALU_TOKEN") else {}

def ok(r):
    if r.status_code >= 400:
        sys.exit(f"FAIL {r.request.method} {r.url}: {r.status_code} {r.text[:300]}")
    return r.json()

print("health", ok(requests.get(f"{API}/api/health")))
d = check_det_dataset("coco8.yaml")                      # downloads 8 images
ds = ok(requests.post(f"{API}/api/datasets/register", headers=H, json={"data_yaml": str(d["yaml_file"])}))
job = ok(requests.post(f"{API}/api/training", headers=H, json={"dataset_id": ds["dataset_id"], "epochs": 10, "batch": 8}))
t0 = time.time()
while True:
    job = ok(requests.get(f"{API}/api/jobs/{job['id']}", headers=H))
    print(f"  {time.time()-t0:5.1f}s {job['status']} {job['progress']} {job['message'][:80]}")
    if job["status"] in ("succeeded", "failed"):
        break
    time.sleep(2)
if job["status"] != "succeeded":
    sys.exit(f"FAIL training: {job['error']}")
model = ok(requests.get(f"{API}/api/models/{job['result_id']}", headers=H))
print("model", {k: model[k] for k in ("id", "metrics", "train_seconds", "onnx", "device")})
img = urllib.request.urlopen("https://ultralytics.com/images/bus.jpg").read()
r = ok(requests.post(f"{API}/api/inference", headers=H, data={"model_id": model["id"], "target_event": "person"},
                     files={"image": ("bus.jpg", img, "image/jpeg")}))
print("trained inference", len(r["detections"]), "dets", r["latency_ms"], "ms, events", r["events"])
zs = ok(requests.post(f"{API}/api/models/zero-shot", headers=H, json={"classes": ["person", "bus"]}))
r = ok(requests.post(f"{API}/api/inference", headers=H, data={"model_id": zs["id"]}, files={"image": ("bus.jpg", img, "image/jpeg")}))
print("zero-shot inference", [(x["cls"], x["score"]) for x in r["detections"]], r["latency_ms"], "ms")
print("E2E OK")
