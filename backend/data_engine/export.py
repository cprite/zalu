"""export_dataset(...) -> DatasetManifest: write a standard Ultralytics YOLO dataset.

Layout (loadable with plain `yolo detect train data=<root>/data.yaml`):
    <root>/data.yaml
    <root>/manifest.json
    <root>/images/{train,val}/<frame_id>.jpg
    <root>/labels/{train,val}/<frame_id>.txt   # "cls cx cy w h", normalised

Data-integrity rules enforced here:
- train/val are split by recording session, never by random frame sampling.
- held-out (independent test) sessions are excluded entirely.
- synthetic images only go to train, and only if their parent real frame is in train.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from .schema import Annotation, AnnotationSet, DatasetManifest, FrameCollection, ManifestImage, SplitStats

TEST_TRAIN_SENTINEL = "DO_NOT_TRAIN_ON_INDEPENDENT_TEST_SET"


class ExportError(ValueError):
    pass


def yolo_line(cls_id: int, xyxy: Sequence[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _place(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _choose_val_sessions(counts: Dict[str, int], val_fraction: float) -> List[str]:
    """Smallest sessions first until val reaches the target; always keep >=1 train session."""
    sessions = sorted(counts, key=lambda s: (counts[s], s))
    if len(sessions) < 2:
        return []
    target = val_fraction * sum(counts.values())
    val, total = [], 0
    for s in sessions[:-1]:
        if total >= target:
            break
        val.append(s)
        total += counts[s]
    return val


def _order_key(a: Annotation, frames: Dict[str, object]):
    f = frames.get(a.frame_id)
    ts = getattr(f, "timestamp_s", None)
    idx = getattr(f, "source_frame_index", None)
    return (getattr(f, "source_file", ""), ts if ts is not None else -1, idx if idx is not None else -1, a.image_path)


def _write(root: Path, split: str, items: List[Annotation], class_names: List[str],
           frames: Dict[str, object], manifest_images: List[ManifestImage]) -> SplitStats:
    st = SplitStats(per_class={c: 0 for c in class_names})
    sessions = set()
    for a in items:
        ext = Path(a.image_path).suffix.lower() or ".jpg"
        img_rel = f"images/{split}/{a.frame_id}{ext}"
        lbl_rel = f"labels/{split}/{a.frame_id}.txt"
        _place(a.image_path, root / img_rel)
        lines = [yolo_line(class_names.index(b.class_name), b.xyxy, a.width, a.height) for b in a.boxes]
        (root / lbl_rel).parent.mkdir(parents=True, exist_ok=True)
        (root / lbl_rel).write_text("\n".join(lines) + ("\n" if lines else ""))
        st.images += 1
        st.boxes += len(a.boxes)
        st.background_images += 0 if a.boxes else 1
        st.synthetic_images += int(a.synthetic)
        for b in a.boxes:
            st.per_class[b.class_name] += 1
        sessions.add(a.session_id)
        f = frames.get(a.frame_id)
        manifest_images.append(ManifestImage(
            frame_id=a.frame_id, split=split, image=img_rel, label=lbl_rel, session_id=a.session_id,
            source_file=getattr(f, "source_file", ""), synthetic=a.synthetic,
            parent_frame_id=getattr(f, "parent_frame_id", None), review_status=a.status,
        ))
    st.sessions = sorted(sessions)
    return st


def _check_classes(aset: AnnotationSet, class_names: List[str]) -> None:
    bad = {b.class_name for a in aset.annotations for b in a.boxes} - set(class_names)
    if bad:
        raise ExportError(f"annotations use classes not in TaskSpec: {sorted(bad)}")


def _frame_index(frames) -> Dict[str, object]:
    if frames is None:
        return {}
    items = frames.frames if isinstance(frames, FrameCollection) else frames
    return {f.frame_id: f for f in items}


def export_dataset(
    annotations: AnnotationSet,
    task,
    out_dir,
    frames: Optional[Iterable] = None,
    val_sessions: Optional[List[str]] = None,
    held_out_sessions: Sequence[str] = (),
    val_fraction: float = 0.2,
    include_synthetic: bool = True,
    dataset_id: Optional[str] = None,
) -> DatasetManifest:
    """Write a YOLO train/val dataset. `frames` (FrameCollection or Frame list, real
    and synthetic) is optional but needed for provenance and synthetic parent checks."""
    class_names = task.class_names
    _check_classes(annotations, class_names)
    fidx = _frame_index(frames)
    root = Path(out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ExportError(f"{root} is not empty; export to a fresh directory")
    held = set(held_out_sessions)
    warnings: List[str] = []

    usable = [a for a in annotations.annotations if a.status != "rejected"]
    real = [a for a in usable if not a.synthetic and a.session_id not in held]
    synth = [a for a in usable if a.synthetic]
    if not real:
        raise ExportError("no usable real frames to export")

    counts: Dict[str, int] = {}
    for a in real:
        counts[a.session_id] = counts.get(a.session_id, 0) + 1

    if val_sessions is not None:
        unknown = set(val_sessions) - set(counts)
        if unknown:
            raise ExportError(f"val_sessions not found among exportable sessions: {sorted(unknown)}")
        if set(val_sessions) & held:
            raise ExportError("a session cannot be both validation and held-out test")
        vset = set(val_sessions)
        strategy = "by_session(manual)"
    else:
        vset = set(_choose_val_sessions(counts, val_fraction))
        strategy = "by_session(auto)"

    if vset:
        train = [a for a in real if a.session_id not in vset]
        val = [a for a in real if a.session_id in vset]
        if not train:
            raise ExportError("all sessions assigned to validation; nothing left to train on")
    else:
        # Single recording: contiguous time blocks with one dropped boundary frame,
        # never a random shuffle of neighbouring frames.
        ordered = sorted(real, key=lambda a: _order_key(a, fidx))
        n_val = max(1, int(round(len(ordered) * val_fraction))) if len(ordered) >= 5 else 0
        cut = len(ordered) - n_val
        train, val = ordered[:max(0, cut - 1)], ordered[cut:] if n_val else []
        strategy = "temporal_block(single_session)"
        warnings.append("only one recording session: validation is a later time block of the same session, "
                        "so val metrics are optimistic. Record a second session for honest validation.")

    train_sessions = {a.session_id for a in train}
    train_ids = {a.frame_id for a in train}
    synth_used: List[Annotation] = []
    if include_synthetic:
        # A synthetic image may enter train only if its parent frame is itself in train. With
        # provenance (frames) we check the exact parent; without it, fall back to the session,
        # which is only safe when val is a different session (not a time block of the same one).
        for a in synth:
            parent = getattr(fidx.get(a.frame_id), "parent_frame_id", None)
            if parent is not None:
                ok = parent in train_ids
            else:
                ok = a.session_id in train_sessions and not strategy.startswith("temporal_block")
            if ok:
                synth_used.append(a)
        dropped = len(synth) - len(synth_used)
        if dropped:
            warnings.append(f"{dropped} synthetic image(s) dropped: parent frame not in train (or provenance missing)")

    images: List[ManifestImage] = []
    splits = {
        "train": _write(root, "train", train + synth_used, class_names, fidx, images),
        "val": _write(root, "val", val, class_names, fidx, images),
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val",
        "names": {i: n for i, n in enumerate(class_names)},
    }, sort_keys=False))

    warnings += _quality_warnings(splits, usable, class_names)
    manifest = DatasetManifest(
        **({"dataset_id": dataset_id} if dataset_id else {}),
        task_id=task.task_id, variant="real_plus_synthetic" if synth_used else "real_only",
        root=str(root), data_yaml=str(data_yaml), class_names=class_names, splits=splits,
        split_strategy=strategy, excluded_sessions=sorted(held), images=images, warnings=warnings,
        annotator=annotations.annotator,
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return manifest


def export_test_set(annotations: AnnotationSet, task, out_dir, sessions: Sequence[str],
                    frames: Optional[Iterable] = None, dataset_id: Optional[str] = None) -> DatasetManifest:
    """Export independent test sessions for the Evaluation Engine (Damir).

    Only real frames from `sessions`. data.yaml's `train` points to a sentinel
    path that does not exist, so accidentally training on it fails loudly.
    Evaluate with: yolo detect val model=best.pt data=<root>/data.yaml split=test
    """
    class_names = task.class_names
    _check_classes(annotations, class_names)
    fidx = _frame_index(frames)
    root = Path(out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ExportError(f"{root} is not empty; export to a fresh directory")
    sset = set(sessions)
    items = [a for a in annotations.annotations
             if a.session_id in sset and a.status != "rejected" and not a.synthetic]
    if not items:
        raise ExportError(f"no real frames found for test sessions {sorted(sset)}")
    images: List[ManifestImage] = []
    splits = {"test": _write(root, "test", items, class_names, fidx, images)}
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root), "train": TEST_TRAIN_SENTINEL, "val": "images/test", "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }, sort_keys=False))
    unreviewed = sum(1 for a in items if a.status != "reviewed")
    warnings = []
    if unreviewed:
        warnings.append(f"{unreviewed}/{len(items)} test images are auto-labelled and not human-reviewed; "
                        "test labels should be reviewed before reporting metrics")
    if len(items) < 50:
        warnings.append(f"only {len(items)} test images: metrics will have wide uncertainty")
    manifest = DatasetManifest(
        **({"dataset_id": dataset_id} if dataset_id else {}),
        task_id=task.task_id, variant="test", root=str(root), data_yaml=str(data_yaml),
        class_names=class_names, splits=splits, split_strategy="independent_sessions",
        images=images, warnings=warnings, annotator=annotations.annotator,
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return manifest


def _quality_warnings(splits: Dict[str, SplitStats], usable: List[Annotation], class_names: List[str]) -> List[str]:
    w: List[str] = []
    tr, va = splits["train"], splits["val"]
    if tr.images < 20:
        w.append(f"only {tr.images} training images")
    if va.images == 0:
        w.append("validation split is empty")
    for c in class_names:
        if tr.per_class.get(c, 0) < 10:
            w.append(f"class '{c}' has only {tr.per_class.get(c, 0)} training boxes")
        if va.images and va.per_class.get(c, 0) == 0:
            w.append(f"class '{c}' does not appear in validation")
    if tr.images and tr.background_images / tr.images > 0.5:
        w.append(f"{tr.background_images}/{tr.images} training images have no boxes")
    pending = sum(1 for a in usable if a.needs_review and a.status == "auto")
    if usable and pending / len(usable) > 0.3:
        w.append(f"{pending}/{len(usable)} frames flagged for review were not reviewed")
    return w
