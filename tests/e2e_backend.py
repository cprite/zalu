"""End-to-end check through the public API only (app on nikopi5 -> GPU worker on GX10).

  ZALU_API=https://nikopi5.tail4aaac7.ts.net:10000 ZALU_TOKEN=... python3 tests/e2e_backend.py [--footage video.mp4]

Steps: health, task (Claude subscription), register coco8, train, poll, inference, zero-shot, ONNX download,
mock evaluation; with --footage also the real Data Engine (upload -> frames -> auto-labels -> dataset).
Needs only `requests`.
"""
import argparse
import os
import sys
import time
import urllib.request

import requests

API = os.environ.get("ZALU_API", "http://127.0.0.1:9100")
H = {"Authorization": f"Bearer {os.environ['ZALU_TOKEN']}"} if os.environ.get("ZALU_TOKEN") else {}


def ok(r):
    if r.status_code >= 400:
        sys.exit(f"FAIL {r.request.method} {r.url}: {r.status_code} {r.text[:400]}")
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--footage", help="video/image file to run through the real Data Engine")
    ap.add_argument("--prompt", default="Detect workers without safety vests")
    a = ap.parse_args()

    h = ok(requests.get(f"{API}/api/health"))
    print("health", h["host"], h["gpu_worker"].get("device"), h["engines"])

    t0 = time.time()
    task = ok(requests.post(f"{API}/api/tasks", headers=H, json={"prompt": a.prompt}, timeout=300))
    print(f"task {task['task_id']} {task['status']} by {task['compiler']} in {time.time()-t0:.1f}s:",
          [c["name"] for c in task["classes"]])

    ds = ok(requests.post(f"{API}/api/datasets/register", headers=H, json={"data_yaml": "coco8.yaml"}, timeout=300))
    print("dataset", ds["dataset_id"], {k: v["images"] for k, v in ds["splits"].items()})
    job = ok(requests.post(f"{API}/api/training", headers=H, json={"dataset_id": ds["dataset_id"], "epochs": 10, "batch": 8}))
    t0 = time.time()
    while job["status"] not in ("succeeded", "failed"):
        time.sleep(2)
        job = ok(requests.get(f"{API}/api/jobs/{job['id']}", headers=H))
    print(f"training {job['status']} in {time.time()-t0:.1f}s wall: {job['message'][:90]}")
    if job["status"] != "succeeded":
        sys.exit(f"FAIL training: {job['error']}")
    model = ok(requests.get(f"{API}/api/models/{job['result_id']}", headers=H))
    print("model", model["id"], "val", model["metrics"], "baseline", model["baseline_metrics"], "onnx check", model["export_check"].get("ok"))

    img = urllib.request.urlopen(urllib.request.Request("https://ultralytics.com/images/bus.jpg",
                                                        headers={"User-Agent": "zalu-e2e"})).read()
    for _ in range(2):
        r = ok(requests.post(f"{API}/api/inference", headers=H, data={"model_id": model["id"], "target_event": "person"},
                             files={"image": ("bus.jpg", img, "image/jpeg")}))
    print("inference", len(r["detections"]), "dets, worker latency", r["latency_ms"], "ms, events", r["events"])
    zs = ok(requests.post(f"{API}/api/models/zero-shot", headers=H, json={"classes": ["person", "bus"]}))
    r = ok(requests.post(f"{API}/api/inference", headers=H, data={"model_id": zs["id"]}, files={"image": ("bus.jpg", img, "image/jpeg")}))
    print("zero-shot", [(x["cls"], x["score"]) for x in r["detections"]])
    onnx = requests.get(f"{API}/api/models/{model['id']}/onnx", headers=H, timeout=120)
    print("onnx download", onnx.status_code, len(onnx.content), "bytes")
    rep = ok(requests.post(f"{API}/api/evaluations", headers=H, json={"model_ids": [model["id"]], "test_dataset_id": ds["dataset_id"]}))
    print("evaluation mock", rep["mock"])

    if a.footage:
        if task["status"] != "ok":
            sys.exit("FAIL: task spec not ok, cannot run the Data Engine")
        t0 = time.time()
        with open(a.footage, "rb") as fh:
            d = ok(requests.post(f"{API}/api/datasets", headers=H, data={"task_id": task["task_id"]},
                                 files={"files": (os.path.basename(a.footage), fh, "video/mp4")}, timeout=1800))
        print(f"data engine {d['dataset_id']} in {time.time()-t0:.1f}s:", {k: (v["images"], v["boxes"], v["per_class"]) for k, v in d["splits"].items()},
              "warnings", d["warnings"][:3])
    print("E2E OK")


if __name__ == "__main__":
    main()
