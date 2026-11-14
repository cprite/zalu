"""Timed user workflow through the public API: prompt -> TaskSpec -> dataset from footage -> training -> ONNX -> inference.
Measures the real wall time of each stage (the '15 minutes' target).

  ZALU_API=... ZALU_TOKEN=... python3 tests/timed_workflow.py --footage video.mp4 [--prompt "..."] [--epochs 40]
"""
import argparse, json, os, sys, time

import requests

API = os.environ.get("ZALU_API", "http://127.0.0.1:9100")
H = {"Authorization": f"Bearer {os.environ['ZALU_TOKEN']}"} if os.environ.get("ZALU_TOKEN") else {}


def ok(r):
    if r.status_code >= 400:
        sys.exit(f"FAIL {r.request.method} {r.url}: {r.status_code} {r.text[:400]}")
    return r.json()


ap = argparse.ArgumentParser()
ap.add_argument("--footage", required=True)
ap.add_argument("--prompt", default="Alert me when a worker is not wearing a hi-vis safety vest")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--fps", type=float, default=1.0)
ap.add_argument("--max-frames", type=int, default=200)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--fast", action="store_true", help="demo mode: no baseline val, no ONNX check")
a = ap.parse_args()
T, t_all = {}, time.time()

t = time.time()
task = ok(requests.post(f"{API}/api/tasks", headers=H, json={"prompt": a.prompt}, timeout=300))
T["1_task_spec"] = time.time() - t
print("task", task["task_id"], task["status"], [c["name"] for c in task["classes"]], flush=True)

t = time.time()
with open(a.footage, "rb") as fh:
    ds = ok(requests.post(f"{API}/api/datasets", headers=H, data={"task_id": task["task_id"], "fps": a.fps, "max_frames": a.max_frames},
                          files={"files": (os.path.basename(a.footage), fh, "video/mp4")}, timeout=1800))
T["2_dataset_upload_frames_autolabel"] = time.time() - t
print("dataset", ds["dataset_id"], {k: (v["images"], v["per_class"]) for k, v in ds["splits"].items()}, flush=True)

t = time.time()
job = ok(requests.post(f"{API}/api/training", headers=H, json={"dataset_id": ds["dataset_id"], "epochs": a.epochs, "batch": 16, "imgsz": a.imgsz,
                                                              "baseline": not a.fast, "verify_export": not a.fast}))
while job["status"] not in ("succeeded", "failed"):
    time.sleep(2)
    job = ok(requests.get(f"{API}/api/jobs/{job['id']}", headers=H))
T["3_training_incl_baseline_onnx"] = time.time() - t
if job["status"] != "succeeded":
    sys.exit(f"FAIL training: {job['error']}")
model = ok(requests.get(f"{API}/api/models/{job['result_id']}", headers=H))
print("model", model["id"], "val", model["metrics"], "onnx_check", model["export_check"], flush=True)

t = time.time()
onnx = requests.get(f"{API}/api/models/{model['id']}/onnx", headers=H, timeout=300)
T["4_onnx_download"] = time.time() - t

# one frame from the dataset's val split, fetched through the API (what the review screen would show)
val_img = next((im for im in ds["images"] if im["split"] == "val"), None)
lat = []
if val_img:
    img = requests.get(f"{API}/api/files", headers=H, params={"path": os.path.join(ds["root"], val_img["image"])}, timeout=60).content
    for _ in range(5):
        t = time.time()
        r = ok(requests.post(f"{API}/api/inference", headers=H, data={"model_id": model["id"], "target_event": task.get("target_event") or ""},
                             files={"image": ("f.jpg", img, "image/jpeg")}))
        lat.append(time.time() - t)
    print("inference", [(d["cls"], d["score"]) for d in r["detections"]], "events", r["events"], flush=True)
T["total_wall"] = time.time() - t_all
res = {"timings_s": {k: round(v, 1) for k, v in T.items()}, "onnx_bytes": len(onnx.content),
       "inference_roundtrip_ms_median": round(sorted(lat)[len(lat) // 2] * 1000) if lat else None,
       "epochs": a.epochs, "prompt": a.prompt}
print(json.dumps(res, indent=1))
