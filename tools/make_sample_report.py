"""Generate the frontend's SAMPLE EvaluationReport with the real Evaluation Engine on invented predictions.

The numbers are NOT experimental results: ground truth and predictions are simulated with a fixed seed so the UI
has a realistic, schema-exact report to render. Every entry is marked sample_data=True.

    python -m tools.make_sample_report   ->  frontend/src/mocks/sample-report.json
"""
import json
import random
from pathlib import Path

from backend import schemas
from backend.evaluation_engine import core
from backend.evaluation_engine.matching import Box

rng = random.Random(2026)
W, H = 1280, 720
CLASSES = ["no_goggles", "goggles"]
SESS = ["sess_lab_bench_2", "sess_lab_door_3"]
images = [f"test/{SESS[k % 2]}/frame_{k:03d}.jpg" for k in range(48)]
sessions = {im: im.split("/")[1] for im in images}

gt = []
for im in images:
    if rng.random() < 0.15:
        continue  # empty frames: nobody in view
    for _ in range(rng.choice([1, 1, 2, 3])):
        x, y, s = rng.uniform(40, W - 240), rng.uniform(40, H - 260), rng.uniform(90, 200)
        gt.append(Box.of(im, rng.choice(CLASSES), [x, y, x + s, y + s * 1.2]))


def simulate(recall, fp_rate, swap, low_conf, jitter):
    preds = []
    for g in gt:
        r = rng.random()
        x1, y1, x2, y2 = g.box
        j = lambda: rng.uniform(-jitter, jitter) * (x2 - x1)
        box = [x1 + j(), y1 + j(), x2 + j(), y2 + j()]
        if r < recall:
            cls = g.cls if rng.random() > swap else [c for c in CLASSES if c != g.cls][0]
            preds.append(Box.of(g.image, cls, box, round(rng.uniform(0.4, 0.97), 3)))
        elif r < recall + low_conf:
            preds.append(Box.of(g.image, g.cls, box, round(rng.uniform(0.06, 0.22), 3)))
    for im in images:
        if rng.random() < fp_rate:
            x, y = rng.uniform(0, W - 150), rng.uniform(0, H - 150)
            preds.append(Box.of(im, rng.choice(CLASSES), [x, y, x + 120, y + 140], round(rng.uniform(0.3, 0.8), 3)))
    return preds


models = {
    "model_sample_zeroshot": dict(kind="zero_shot", variant=None, base_model="yoloe-11l-seg.pt",
                                  sim=(0.62, 0.35, 0.25, 0.15, 0.12), lat=16.4),
    "model_sample_real": dict(kind="trained", variant="real_only", base_model="yolo11n.pt",
                              sim=(0.88, 0.12, 0.05, 0.06, 0.06), lat=4.1),
    "model_sample_synth": dict(kind="trained", variant="real_plus_synthetic", base_model="yolo11n.pt",
                               sim=(0.87, 0.10, 0.05, 0.06, 0.06), lat=4.1),
}
per_model, meta = {}, {}
for mid, m in models.items():
    preds = simulate(*m["sim"])
    lat = [round(m["lat"] + rng.uniform(-0.8, 1.5), 2) for _ in images]
    e = core.evaluate_model(gt, preds, images, CLASSES, 0.5, 0.25, sessions=sessions, latencies_ms=lat)
    e["model"] = {"id": mid, "kind": m["kind"], "variant": m["variant"], "base_model": m["base_model"],
                  "dataset_id": None if m["kind"] == "zero_shot" else f"ds_sample_{m['variant']}",
                  "epochs": None if m["kind"] == "zero_shot" else 30, "created_at": 1794700000.0,
                  "val_metrics": {} if m["kind"] == "zero_shot" else {"precision": 0.91, "recall": 0.9}}
    e["test"].update({"dataset_id": "ds_sample_test", "split": "test"})
    e["sample_data"] = True
    for f in e["failures"]:
        f["thumbnail"] = None
    per_model[mid] = e
    meta[mid] = {"kind": m["kind"], "variant": m["variant"]}

cmp = core.compare(per_model, meta)
req = schemas.EvaluationRequest(model_ids=list(models), test_dataset_id="ds_sample_test")
rep = schemas.EvaluationReport(
    id="rep_sample", request=req, per_model=per_model,
    notes=["SAMPLE DATA: simulated predictions for UI development, not an experimental result.",
           f"Test split 'test' of dataset ds_sample_test: {len(images)} images.",
           "Matching at IoU >= 0.5, confidence >= 0.25 (fixed before testing)."] + cmp["notes"],
)
out = Path(__file__).resolve().parents[1] / "frontend" / "src" / "mocks" / "sample-report.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({**rep.model_dump(), "image_size": [W, H]}, indent=1))
for mid, e in per_model.items():
    print(mid, e["overall"]["precision"], e["overall"]["recall"], e["overall"]["f1"], e["readiness"]["verdict"])
print(cmp.get("synthetic_ab"), "->", out)
