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
from backend.evaluation_engine.yolo import parse_yolo_labels


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
    unreviewed: int = 0


def read_bytes(path: str) -> bytes:
    p = Path(path)
    if p.exists():
        return p.read_bytes()
    from backend.infrastructure import gpu_client  # remote: the file lives on the GX10

    response = gpu_client.stream_file(path)
    try:
        return response.content
    finally:
        response.close()


def _annotation_set(ds) -> Optional[dict]:
    local = Path(ds.root).parent / "work" / "annotations.json"
    if local.exists():
        import json

        return json.loads(local.read_text())
    from backend.infrastructure import gpu_client

    try:
        return gpu_client.post("/datasets/annotations", json={"manifest": ds.model_dump()}, timeout=120)
    except gpu_client.GPUWorkerError as exc:
        # Older GX10 workers return 500 for manifests registered without a work directory.
        if exc.status in (404, 500) and not ds.annotator:
            return None
        raise


def image_size(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except ImportError:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode evaluation image")
        h, w = frame.shape[:2]
        return w, h
    except OSError as exc:
        raise ValueError("could not decode evaluation image") from exc


def load_test_set(ds, reader: Callable[[str], bytes] = read_bytes, annotations: Optional[dict] = None) -> TestSet:
    """Load only the final test split."""
    return load_split(ds, "test", reader=reader, annotations=annotations)


def load_split(ds, split: str, *, reader: Callable[[str], bytes] = read_bytes,
               annotations: Optional[dict] = None) -> TestSet:
    """Load exactly the requested manifest split, with no implicit fallback."""
    if split not in ("test", "val"):
        raise ValueError("evaluation split must be test or val")
    images = [im for im in ds.images if im.split == split]
    if not images:
        raise ValueError(f"dataset {ds.dataset_id} has no '{split}' frame roster")
    if len({im.image for im in images}) != len(images) or len({im.frame_id for im in images}) != len(images):
        raise ValueError(f"dataset {ds.dataset_id} has duplicate image paths or frame IDs")
    root = Path(ds.root)
    frames = [TestFrame(im.frame_id, im.image, str(root / im.image), im.session_id, bool(im.synthetic)) for im in images]
    names = list(ds.class_names)

    aset = annotations if annotations is not None else _annotation_set(ds)
    gt: list[Box] = []
    if aset:
        if aset.get("class_names", names) != names:
            raise ValueError("annotation class order differs from the dataset manifest")
        by_id = {a["frame_id"]: a for a in aset["annotations"]}
        if len(by_id) != len(aset["annotations"]):
            raise ValueError("duplicate frame IDs in the annotation set")
        kept = []
        unreviewed = 0
        for f in frames:
            a = by_id.get(f.frame_id)
            if a is None:
                raise ValueError(f"{f.image}: missing annotation for manifest frame")
            if a.get("status") == "rejected":
                continue
            if a.get("session_id", f.session_id) != f.session_id or a.get("synthetic", f.synthetic) != f.synthetic:
                raise ValueError(f"{f.image}: annotation provenance differs from the manifest")
            kept.append(f)
            unreviewed += a.get("status") != "reviewed"
            for b in a["boxes"]:
                if b["class_name"] not in names:
                    raise ValueError(f"{f.image}: undeclared class {b['class_name']!r}")
                gt.append(Box.of(f.image, b["class_name"], b["xyxy"]))
        return TestSet(split, kept, gt, names, "annotations", unreviewed)

    # Registered dataset: ground truth lives in five-column YOLO detection files.
    label_of = {im.frame_id: str(root / im.label) for im in images}
    for f in frames:
        path = label_of[f.frame_id]
        try:
            text = reader(path).decode("utf-8")
        except (FileNotFoundError, UnicodeError) as exc:
            raise ValueError(f"{f.image}: cannot read YOLO labels at {path}") from exc
        if text.strip():
            gt.extend(parse_yolo_labels(f.image, text, image_size(reader(f.abs_path)), names, label_path=path))
    return TestSet(split, frames, gt, names, "yolo_labels",
                   sum(im.review_status != "reviewed" for im in images))


def train_sessions(ds) -> set[str]:
    """Recording sessions a training dataset used (train + val frames)."""
    s = {im.session_id for im in ds.images if im.split in ("train", "val") and im.session_id}
    for st in (ds.splits or {}).values():
        s |= set(getattr(st, "sessions", []) or [])
    return s
