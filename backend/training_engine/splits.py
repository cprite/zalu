"""Train / val / test separation check for a YOLO data.yaml.

Rule from the brief: test data must never enter training. We check that no image file (by resolved path and by
content hash) appears in more than one split. Splitting by recording session is the Data Engine's job; this is
the backend's last line of defence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(root: Path, entry) -> list[Path]:
    if entry is None:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    out: list[Path] = []
    for e in entries:
        p = (root / e) if not Path(e).is_absolute() else Path(e)
        if p.is_dir():
            out += [q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT]
        elif p.suffix == ".txt" and p.exists():  # list file
            out += [(root / l.strip()) if not Path(l.strip()).is_absolute() else Path(l.strip())
                    for l in p.read_text().splitlines() if l.strip()]
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
    return [q.resolve() for q in out]


def resolve(data_yaml: str) -> dict:
    """Absolute split paths the same way the trainer sees them (Ultralytics resolves 'path' against its
    datasets dir and downloads sample datasets such as coco8)."""
    try:
        from ultralytics.data.utils import check_det_dataset

        return check_det_dataset(data_yaml)
    except Exception:
        cfg = yaml.safe_load(Path(data_yaml).read_text())
        root = Path(cfg.get("path") or Path(data_yaml).parent)
        cfg["path"] = str(root if root.is_absolute() else (Path(data_yaml).parent / root).resolve())
        return cfg


def _hash(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def check_splits(data_yaml: str, variant: str = "real_only") -> dict[str, Any]:
    """variant="test": an independent test set (Kirill's export points val at the test images so YOLO can
    validate on it). It must contain NO train images; val == test is expected there."""
    if variant == "test":
        cfg = resolve(data_yaml)
        root = Path(cfg.get("path") or Path(data_yaml).parent)
        tr = _images(root, cfg.get("train")) if cfg.get("train") else []
        te = _images(root, cfg.get("test") or cfg.get("val"))
        errors = (["test dataset contains train images"] if tr and set(tr) - set(te) else []) + ([] if te else ["test split is empty"])
        return {"counts": {"train": 0, "val": 0, "test": len(te)}, "errors": errors, "warnings": [], "ok": not errors}
    cfg = resolve(data_yaml)
    root = Path(cfg.get("path") or Path(data_yaml).parent)
    splits = {k: _images(root, cfg.get(k)) for k in ("train", "val", "test")}
    counts = {k: len(v) for k, v in splits.items()}
    errors: list[str] = []
    if counts["train"] == 0:
        errors.append("train split is empty")
    if counts["val"] == 0:
        errors.append("val split is empty")
    hashes = {k: {_hash(p): p for p in v} for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        same_path = set(splits[a]) & set(splits[b])
        same_bytes = set(hashes[a]) & set(hashes[b])
        n = len(same_path | {hashes[a][h] for h in same_bytes})
        if n and not (a == "train" and b == "val" and cfg.get("allow_train_val_overlap")):
            errors.append(f"{n} image(s) shared between {a} and {b}")
    warnings = [] if counts["test"] else ["no test split: independent evaluation needs a test set (Data Engine)"]
    return {"counts": counts, "errors": errors, "warnings": warnings, "ok": not errors}
