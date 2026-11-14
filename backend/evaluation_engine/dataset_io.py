"""Load the independent test set for evaluation (owner: Damir).

Works with Kirill's contracts (backend/data_engine/schema.py): DatasetManifest.images carries split, session_id and
synthetic per frame; ground truth comes from the Data Engine's AnnotationSet (reviewed labels), or, for datasets that
were only registered (no Data Engine work dir), from the YOLO label files.

Files are read locally when they exist on this host, otherwise through the GX10 GPU worker (backend.infrastructure
.gpu_client), so the hook works both on nikopi5 (API host) and on the GX10.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from backend.evaluation_engine.matching import Box


@dataclass
class TestFrame:
    frame_id: str
    image: str          # path relative to the dataset root (stable id in reports)
    abs_path: str       # absolute path on the host that owns the files
    session_id: str
    synthetic: bool


@dataclass
class TestSet:
    split: str
    frames: list[TestFrame]
    gt: list[Box]
    names: list[str]
    source: str         # "annotations" | "yolo_labels"


def read_bytes(path: str) -> bytes:
    p = Path(path)
    if p.exists():
        return p.read_bytes()
    from backend.infrastructure import gpu_client  # remote: the file lives on the GX10

    return gpu_client.stream_file(path).content


def _annotation_set(ds) -> Optional[dict]:
    local = Path(ds.root).parent / "work" / "annotations.json"
    if local.exists():
        import json

        return json.loads(local.read_text())
    try:
        from backend.infrastructure import gpu_client

        return gpu_client.post("/datasets/annotations", json={"manifest": ds.model_dump()}, timeout=120)
    except Exception:  # registered datasets have no Data Engine work dir
        return None


def _image_size(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except ImportError:
        import cv2
        import numpy as np

        h, w = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        return w, h


def load_test_set(ds, reader: Callable[[str], bytes] = read_bytes, annotations: Optional[dict] = None) -> TestSet:
    """Frames of the 'test' split (falls back to 'val' with a warning upstream) plus pixel-space ground truth."""
    by_split: dict[str, list] = {}
    for im in ds.images:
        by_split.setdefault(im.split, []).append(im)
    split = "test" if by_split.get("test") else ("val" if by_split.get("val") else next(iter(by_split), "test"))
    images = by_split.get(split, [])
    root = Path(ds.root)
    frames = [TestFrame(im.frame_id, im.image, str(root / im.image), im.session_id, bool(im.synthetic)) for im in images]
    names = list(ds.class_names)

    aset = annotations if annotations is not None else _annotation_set(ds)
    gt: list[Box] = []
    if aset:
        wanted = {f.frame_id: f for f in frames}
        for a in aset.get("annotations", []):
            f = wanted.get(a["frame_id"])
            if f is None or a.get("status") == "rejected":
                continue
            for b in a.get("boxes", []):
                gt.append(Box.of(f.image, b["class_name"], b["xyxy"]))
        frames = [f for f in frames if not any(a["frame_id"] == f.frame_id and a.get("status") == "rejected"
                                               for a in aset.get("annotations", []))]
        return TestSet(split, frames, gt, names, "annotations")

    # registered YOLO dataset: normalised "cls cx cy w h" label files next to the images
    label_of = {im.frame_id: str(root / im.label) for im in images}
    for f in frames:
        try:
            text = reader(label_of[f.frame_id]).decode()
        except Exception:
            continue  # no label file = frame with nothing to detect
        lines = [l.split() for l in text.splitlines() if l.strip()]
        if not lines:
            continue
        w, h = _image_size(reader(f.abs_path))
        for p in lines:
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            gt.append(Box.of(f.image, names[c], [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return TestSet(split, frames, gt, names, "yolo_labels")


def train_sessions(ds) -> set[str]:
    """Recording sessions a training dataset used (train + val frames)."""
    s = {im.session_id for im in ds.images if im.split in ("train", "val") and im.session_id}
    for st in (ds.splits or {}).values():
        s |= set(getattr(st, "sessions", []) or [])
    return s
