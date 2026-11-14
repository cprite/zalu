"""Deployment Engine: load a model once, run inference on a frame, measure latency, fire simple event rules."""
from __future__ import annotations

import threading
import time
from functools import lru_cache

import numpy as np

from backend import schemas

_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load(model_key: str, weights: str, classes: tuple[str, ...], zero_shot: bool):
    from ultralytics import YOLO

    m = YOLO(weights)
    if zero_shot:
        m.set_classes(list(classes), m.get_text_pe(list(classes)))
    return m


def predict(
    art: schemas.ModelArtifact,
    image: np.ndarray,
    conf: float = 0.25,
    target_event: str | None = None,
) -> schemas.InferenceResult:
    zero_shot = art.kind == "zero_shot"
    weights = art.base_model if zero_shot else art.weights
    if not weights:
        raise ValueError(f"model {art.id} has no weights")
    with _lock:  # one GPU call at a time keeps latency numbers honest
        m = _load(art.id, weights, tuple(art.classes), zero_shot)
        t0 = time.perf_counter()
        r = m.predict(image, conf=conf, verbose=False, device=0)[0]
        latency = (time.perf_counter() - t0) * 1000
    names = art.classes if zero_shot else r.names
    dets = [
        schemas.Detection(cls=str(names[int(c)]), score=round(float(s), 3), box=[round(float(v), 1) for v in b])
        for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]
    events = [target_event] if target_event and any(d.cls == target_event for d in dets) else []
    h, w = image.shape[:2]
    return schemas.InferenceResult(
        model_id=art.id, width=w, height=h, detections=dets, latency_ms=round(latency, 1), events=events
    )
