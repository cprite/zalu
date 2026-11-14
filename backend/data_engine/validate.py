"""Standalone YOLO dataset validator (no VisionForge backend required).

    python -m backend.data_engine validate path/to/dataset
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml
from pydantic import BaseModel

from .ingest import IMAGE_EXT, dhash, hamming


class ValidationReport(BaseModel):
    ok: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, dict]


def _resolve(base: Path, entry: str) -> Path:
    p = Path(entry)
    return p if p.is_absolute() else base / p


def validate_dataset(path, max_errors: int = 50) -> ValidationReport:
    path = Path(path)
    yaml_path = path / "data.yaml" if path.is_dir() else path
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, dict] = {}

    def err(msg):
        if len(errors) < max_errors:
            errors.append(msg)

    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as e:  # noqa: BLE001
        return ValidationReport(ok=False, errors=[f"cannot read {yaml_path}: {e}"], warnings=[], stats={})

    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        err("data.yaml: 'names' missing or empty")
        names = {}
    if sorted(int(k) for k in names) != list(range(len(names))):
        err("data.yaml: class ids must be 0..N-1")
    nc = len(names)
    if "nc" in data and data["nc"] != nc:
        err(f"data.yaml: nc={data['nc']} but {nc} names")

    base = _resolve(yaml_path.parent, data.get("path", "."))
    seen: Dict[str, str] = {}
    dirs_done: Dict[Path, str] = {}
    for split in ("train", "val", "test"):
        entry = data.get(split)
        if not entry:
            if split in ("train", "val"):
                err(f"data.yaml: '{split}' missing")
            continue
        img_dir = _resolve(base, entry)
        if not img_dir.is_dir():
            if split == "train" and "DO_NOT_TRAIN" in str(entry):
                continue  # test-only dataset
            err(f"{split}: image dir {img_dir} does not exist")
            continue
        if img_dir.resolve() in dirs_done:  # e.g. test-only datasets use val == test
            stats[split] = {"alias_of": dirs_done[img_dir.resolve()]}
            continue
        dirs_done[img_dir.resolve()] = split
        # Ultralytics convention: replace the last "images" path component with "labels".
        parts = list(img_dir.parts)
        idx = len(parts) - 1 - parts[::-1].index("images") if "images" in parts else None
        if idx is None:
            err(f"{split}: image dir {img_dir} has no 'images' component; labels cannot be located")
            continue
        parts[idx] = "labels"
        lbl_dir = Path(*parts)
        s = dict(images=0, labels_missing=0, background=0, boxes=0, per_class={int(k): 0 for k in names})
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXT:
                continue
            s["images"] += 1
            if img.name in seen and seen[img.name] != split:
                err(f"{img.name} appears in both {seen[img.name]} and {split}")
            seen[img.name] = split
            im = cv2.imread(str(img))
            if im is None:
                err(f"{split}/{img.name}: unreadable image")
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                s["labels_missing"] += 1
                continue
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            if not lines:
                s["background"] += 1
            for n, ln in enumerate(lines, 1):
                parts = ln.split()
                if len(parts) != 5:
                    err(f"{lbl.name}:{n}: expected 5 fields, got {len(parts)}")
                    continue
                try:
                    c = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    err(f"{lbl.name}:{n}: non-numeric value")
                    continue
                if not 0 <= c < nc:
                    err(f"{lbl.name}:{n}: class id {c} outside 0..{nc - 1}")
                    continue
                if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                    err(f"{lbl.name}:{n}: coordinates not normalised: {parts[1:]}")
                elif cx - w / 2 < -1e-3 or cx + w / 2 > 1 + 1e-3 or cy - h / 2 < -1e-3 or cy + h / 2 > 1 + 1e-3:
                    err(f"{lbl.name}:{n}: box extends outside image")
                s["boxes"] += 1
                s["per_class"][c] += 1
        if s["images"] == 0:
            err(f"{split}: no images")
        if s["labels_missing"]:
            warnings.append(f"{split}: {s['labels_missing']} images without a label file (treated as background)")
        s["per_class"] = {names.get(k, str(k)): v for k, v in s["per_class"].items()}
        stats[split] = s

    _check_manifest(yaml_path.parent / "manifest.json", errors, warnings)
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings, stats=stats)


def _check_manifest(mpath: Path, errors: List[str], warnings: List[str]) -> None:
    if not mpath.exists():
        warnings.append("no manifest.json: session separation cannot be verified")
        return
    m = json.loads(mpath.read_text())
    by_split: Dict[str, set] = {}
    for im in m.get("images", []):
        if not im.get("synthetic"):
            by_split.setdefault(im["split"], set()).add(im["session_id"])
        elif im["split"] != "train":
            errors.append(f"synthetic image {im['frame_id']} in {im['split']} split")
    splits = list(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = by_split[a] & by_split[b]
            if shared and m.get("split_strategy", "").startswith("by_session"):
                errors.append(f"sessions shared between {a} and {b}: {sorted(shared)}")
    present = set().union(*by_split.values()) if by_split else set()
    leaked = set(m.get("excluded_sessions", [])) & present
    if leaked:
        errors.append(f"held-out sessions present in dataset: {sorted(leaked)}")


def check_leakage(train_root, test_root, max_distance: int = 4) -> List[str]:
    """Compare a training dataset with an independent test set.

    Returns problems: shared session ids, or near-identical images (dHash).
    """
    problems: List[str] = []
    tm = json.loads((Path(train_root) / "manifest.json").read_text())
    sm = json.loads((Path(test_root) / "manifest.json").read_text())
    shared = {i["session_id"] for i in tm["images"]} & {i["session_id"] for i in sm["images"]}
    if shared:
        problems.append(f"sessions in both train dataset and test set: {sorted(shared)}")

    def hashes(root, man) -> Dict[str, int]:
        out = {}
        for i in man["images"]:
            im = cv2.imread(str(Path(root) / i["image"]))
            if im is not None:
                out[i["frame_id"]] = dhash(im)
        return out

    th, sh = hashes(train_root, tm), hashes(test_root, sm)
    for sid, s in sh.items():
        for tid, t in th.items():
            if hamming(s, t) <= max_distance:
                problems.append(f"test image {sid} near-identical to train image {tid}")
                break
    return problems
