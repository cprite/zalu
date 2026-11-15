"""The hook the shared API calls: evaluate(req, models, test) -> EvaluationReport (owner: Damir).

Steps: load the independent test split (Kirill's manifest + AnnotationSet) -> leakage checks -> run every model on
every test frame through the same inference path as production (GX10 worker /infer, or the local Deployment Engine
when ZALU_EVAL_LOCAL=1 on the GPU host) -> match, score, compare -> Failure Gallery thumbnails.
"""
from __future__ import annotations

import base64
import io
import os
from collections.abc import Mapping
from math import isfinite
from typing import Callable, Optional

from backend import schemas
from backend.evaluation_engine import core, dataset_io, leakage
from backend.evaluation_engine.yolo import from_inference_result

# Predictions are collected at a low confidence so misses can be explained ("seen at 0.18, below 0.25").
COLLECT_CONF = 0.05
IOU = float(os.environ.get("ZALU_EVAL_IOU", core.DEFAULT_IOU))
CONF = float(os.environ.get("ZALU_EVAL_CONF", core.DEFAULT_CONF))
THUMB_PX = 360

PredictFn = Callable[[schemas.ModelArtifact, bytes, float], dict]


def predict_remote(art: schemas.ModelArtifact, image: bytes, conf: float) -> dict:
    """Inference on the GX10 worker, exactly like POST /api/inference."""
    from backend.infrastructure import gpu_client

    return gpu_client.post("/infer", data={"model": art.model_dump_json(), "conf": conf},
                           files={"image": ("frame.jpg", image, "image/jpeg")}, timeout=60)


def predict_local(art: schemas.ModelArtifact, image: bytes, conf: float) -> dict:
    import cv2
    import numpy as np

    from backend.deployment_engine import inference as inf

    frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return inf.predict(art, frame, conf=conf).model_dump()


def evaluate(
    req: schemas.EvaluationRequest,
    models: list[schemas.ModelArtifact],
    test: schemas.DatasetManifest,
    iou_threshold: float = IOU,
    conf_threshold: float = CONF,
    criteria: core.Criteria = core.Criteria(),
    predict_fn: Optional[PredictFn] = None,
    reader: Callable[[str], bytes] = dataset_io.read_bytes,
    annotations: Optional[dict] = None,
    class_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> schemas.EvaluationReport:
    from backend.infrastructure import storage

    predict_fn = predict_fn or (predict_local if os.environ.get("ZALU_EVAL_LOCAL") == "1" else predict_remote)
    if not 0 < iou_threshold <= 1 or not 0 <= conf_threshold <= 1:
        raise ValueError("IoU must be in (0, 1] and confidence in [0, 1]")
    if req.test_dataset_id != test.dataset_id or not models or len(req.model_ids) != len(set(req.model_ids)):
        raise ValueError("expected the requested test dataset and unique model IDs")
    if [art.id for art in models] != req.model_ids:
        raise ValueError("model artifacts differ from the requested models")
    ts = dataset_io.load_test_set(test, reader=reader, annotations=annotations)
    train_sets = [storage.load("datasets", m.dataset_id, schemas.DatasetManifest) for m in models if m.dataset_id]
    errors, warns = leakage.check_test_set(test, train_sets, ts.frames, ts.split)

    if not ts.frames:
        raise ValueError("test set has no usable frames")
    provenance_errors = []
    if test.variant != "test":
        provenance_errors.append("The manifest was not exported as an independent test dataset.")
    if ts.unreviewed:
        provenance_errors.append(f"{ts.unreviewed} test frames have no reviewed ground truth.")
    if any(not f.session_id for f in ts.frames):
        provenance_errors.append("Recording session IDs are missing.")
    if any(tr is None for tr in train_sets) or any(m.kind == "trained" and not m.dataset_id for m in models):
        provenance_errors.append("At least one training dataset manifest is missing.")

    images = [f.image for f in ts.frames]
    sessions = {f.image: f.session_id for f in ts.frames if f.session_id} or None
    notes = [f"Test split '{ts.split}' of dataset {test.dataset_id}: {len(images)} images, ground truth from {ts.source}."]
    warns += provenance_errors
    notes += [f"LEAKAGE: {e}" for e in errors] + [f"Warning: {w}" for w in warns]
    notes.append(f"Matching at IoU >= {iou_threshold}, confidence >= {conf_threshold} (fixed before testing).")

    data: dict[str, bytes] = {}
    sizes: dict[str, tuple[int, int]] = {}
    class_maps = class_maps or {}
    per_model, meta = {}, {}
    for art in models:
        preds, lat = [], []
        for f in ts.frames:
            if f.image not in data:
                data[f.image] = reader(f.abs_path)
                sizes[f.image] = dataset_io.image_size(data[f.image])
            r = predict_fn(art, data[f.image], min(COLLECT_CONF, conf_threshold))
            r = r if isinstance(r, dict) else r.model_dump()
            latency = float(r.get("latency_ms") or 0)
            if not isfinite(latency) or latency < 0:
                raise ValueError(f"{f.image}: invalid inference latency")
            lat.append(latency)
            preds.extend(from_inference_result(
                f.image, r, size=sizes[f.image], class_map=class_maps.get(art.id), model_id=art.id,
            ))
        e = core.evaluate_model(ts.gt, preds, images, ts.names, iou_threshold, conf_threshold, sessions=sessions,
                                latencies_ms=lat[1:] or lat,  # first call includes model load
                                criteria=criteria)
        e["model"] = {"id": art.id, "kind": art.kind, "variant": art.variant, "base_model": art.base_model,
                      "dataset_id": art.dataset_id, "epochs": art.epochs, "created_at": art.created_at,
                      "val_metrics": art.metrics}
        e["test"].update({"dataset_id": test.dataset_id, "split": ts.split})
        e["warnings"] = warns + e["warnings"]
        if errors or provenance_errors:
            e["readiness"] = {**e["readiness"], "verdict": "invalid_test_set",
                              "summary": "Final test eligibility failed. " + " ".join(errors + provenance_errors)}
        path_of = {f.image: f.abs_path for f in ts.frames}
        for fl in e["failures"]:
            fl["image_path"] = path_of.get(fl["image"])  # fetchable with GET /api/files?path=
            fl["width"], fl["height"] = sizes[fl["image"]]
            fl["thumbnail"] = _thumb(data.get(fl["image"]), fl)
        e["sample_data"] = False
        per_model[art.id] = e
        meta[art.id] = {"kind": art.kind, "variant": art.variant}

    if not errors and not provenance_errors:
        notes += core.compare(per_model, meta).get("notes", [])
    notes.append("Validation metrics (from training) are for model selection; only these test metrics are reported.")
    return schemas.EvaluationReport(request=req, per_model=per_model, notes=notes)


def _thumb(image: Optional[bytes], f: dict) -> Optional[str]:
    """Small JPEG data URI with the failure drawn on it (red = wrong box, amber = missed object). Optional:
    returns None when no imaging library is installed on this host."""
    if not image:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        im = Image.open(io.BytesIO(image)).convert("RGB")
    except Exception:
        return None
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 300)
    d.rectangle(f["box"], outline=(196, 61, 75) if f["kind"] == "false_positive" else (180, 109, 18), width=w)
    if f.get("predicted_box"):
        d.rectangle(f["predicted_box"], outline=(196, 61, 75), width=max(1, w // 2))
    im.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
