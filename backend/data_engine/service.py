"""Connectors called by the FastAPI backend (backend/main.py hooks).

    process_upload(task, files)          -> DatasetManifest   POST /api/datasets
    process_test_upload(task, files)     -> DatasetManifest   independent test set for evaluation
    save_corrections(ds, labels)         -> dict              POST /api/datasets/{id}/labels
    get_annotations(ds)                  -> AnnotationSet     for the review screen

On-disk layout (DATA_ROOT = $ZALU_DATA, same root as the backend's storage):
    <DATA_ROOT>/dataset_files/<dataset_id>/work/     frames, frames.json, annotations.json, task_spec.json, settings
    <DATA_ROOT>/dataset_files/<dataset_id>/dataset/  YOLO dataset (manifest.root); data.yaml is what training uses
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

from .annotate import auto_annotate
from .corrections import apply_corrections
from .export import export_dataset, export_test_set
from .ingest import process_media
from .preview import render_previews
from .schema import AnnotationSet, DatasetManifest, FrameCollection, LabelCorrection, new_id

log = logging.getLogger(__name__)

_detector = None


def data_root() -> Path:
    return Path(os.environ.get("ZALU_DATA", Path.home() / ".cache" / "visionforge" / "data")).resolve()


def set_detector(detector) -> None:
    """Override the auto-labelling detector (tests, or a pre-loaded model)."""
    global _detector
    _detector = detector


@contextmanager
def _timed(timings: dict, stage: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(time.perf_counter() - t0, 2)


def _get_detector():
    global _detector
    if _detector is None:  # loaded once per process; YOLO-World load takes a few seconds
        from .annotate import YoloWorldDetector
        _detector = YoloWorldDetector()
    return _detector


def _dirs(dataset_id: str):
    base = data_root() / "dataset_files" / dataset_id
    return base, base / "work", base / "dataset"


def _work_dir(ds: DatasetManifest) -> Path:
    work = Path(ds.root).parent / "work"
    if not (work / "annotations.json").exists():
        raise FileNotFoundError(f"dataset {ds.dataset_id} was not produced by the Data Engine (no {work})")
    return work


def _ingest_and_label(task, files: Sequence[str], dataset_id: str, fps: float, max_frames: int, timings: dict):
    if task.status != "ok":
        raise ValueError(f"TaskSpec {task.task_id} has status '{task.status}': {task.issues}")
    _, work, out = _dirs(dataset_id)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    work.mkdir(parents=True, exist_ok=True)
    (work / "task_spec.json").write_text(task.model_dump_json(indent=2))
    with _timed(timings, "ingest"):
        frames = process_media(files, work, task=task, fps=fps, max_frames=max_frames)
    if not frames.frames:
        reasons = sorted({r.reason for r in frames.rejected})
        raise ValueError(f"no usable frames in the upload (rejected: {reasons})")
    with _timed(timings, "detector_load"):
        detector = _get_detector()  # ~0 after the first call in this process
    with _timed(timings, "auto_label"):  # includes prompt calibration
        aset = auto_annotate(frames, task, detector=detector, out_path=str(work / "annotations.json"))
    with _timed(timings, "previews"):
        render_previews(aset, work / "preview")
    return frames, aset, work, out


def process_upload(task, files: List[str], held_out_sessions: Sequence[str] = (),
                   val_sessions: Optional[List[str]] = None, fps: float = 1.0,
                   max_frames: int = 200) -> DatasetManifest:
    """Footage -> frames -> auto-labels -> session-split YOLO train/val dataset."""
    dataset_id, timings, t0 = new_id("ds"), {}, time.perf_counter()
    frames, aset, work, out = _ingest_and_label(task, files, dataset_id, fps, max_frames, timings)
    settings = {"kind": "train", "held_out_sessions": list(held_out_sessions), "val_sessions": val_sessions}
    (work / "settings.json").write_text(json.dumps(settings, indent=2))
    return _finish(_export(task, aset, frames, out, dataset_id, settings), timings, t0)


def process_test_upload(task, files: List[str], fps: float = 1.0, max_frames: int = 200) -> DatasetManifest:
    """Independent test recordings -> test-only dataset (never for training). Labels should be reviewed."""
    dataset_id, timings, t0 = new_id("ds"), {}, time.perf_counter()
    frames, aset, work, out = _ingest_and_label(task, files, dataset_id, fps, max_frames, timings)
    settings = {"kind": "test", "sessions": sorted(frames.sessions)}
    (work / "settings.json").write_text(json.dumps(settings, indent=2))
    return _finish(_export(task, aset, frames, out, dataset_id, settings), timings, t0)


def _finish(manifest: DatasetManifest, timings: dict, t0: float) -> DatasetManifest:
    timings["export"] = round(time.perf_counter() - t0 - sum(timings.values()), 2)
    timings["total"] = round(time.perf_counter() - t0, 2)
    manifest = manifest.model_copy(update={"timings": timings})
    (Path(manifest.root) / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    log.info("dataset %s timings: %s", manifest.dataset_id, timings)
    return manifest


def _export(task, aset, frames, out: Path, dataset_id: str, settings: dict) -> DatasetManifest:
    if settings["kind"] == "test":
        return export_test_set(aset, task, out, settings["sessions"], frames=frames, dataset_id=dataset_id)
    return export_dataset(aset, task, out, frames=frames, dataset_id=dataset_id,
                          held_out_sessions=settings.get("held_out_sessions") or (),
                          val_sessions=settings.get("val_sessions"))


def get_annotations(ds: DatasetManifest) -> AnnotationSet:
    return AnnotationSet.model_validate_json((_work_dir(ds) / "annotations.json").read_text())


def save_corrections(ds: DatasetManifest, labels: List[LabelCorrection]) -> dict:
    """Apply review-screen corrections and rebuild the YOLO files in place (same dataset_id).

    Returns {"dataset_id", "applied", "manifest"}; the caller should store the returned
    manifest, since splits, counts and warnings change after corrections.
    """
    from backend.task_engine import TaskSpec

    work = _work_dir(ds)
    task = TaskSpec.model_validate_json((work / "task_spec.json").read_text())
    frames = FrameCollection.model_validate_json((work / "frames.json").read_text())
    settings = json.loads((work / "settings.json").read_text())
    aset = apply_corrections(get_annotations(ds), labels)  # raises CorrectionError on bad input

    out = Path(ds.root)
    tmp = out.with_name("dataset.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    manifest = _export(task, aset, frames, tmp, ds.dataset_id, settings)
    # Swap directories, then fix the absolute paths that pointed at the temp dir.
    old = out.with_name("dataset.old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    data = yaml.safe_load((out / "data.yaml").read_text())
    data["path"] = str(out)
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    manifest = manifest.model_copy(update={"root": str(out), "data_yaml": str(out / "data.yaml"), "timings": ds.timings})
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (work / "annotations.json").write_text(aset.model_dump_json(indent=2))
    return {"dataset_id": ds.dataset_id, "applied": len(labels), "manifest": manifest.model_dump()}


def add_synthetic(ds: DatasetManifest, synthetic_dirs: Optional[Sequence[str]] = None) -> DatasetManifest:
    """Build the real+synthetic variant of a train dataset for the A/B experiment.

    Same real frames, same labels, same val split as `ds` (val sessions are pinned), plus the
    kept synthetic images from `synthetic_dirs` (default: every <work>/synthetic_* batch), train only.
    Returns a NEW dataset (new dataset_id, variant "real_plus_synthetic"); `ds` is untouched.
    Corrections made to `ds` afterwards are not copied over: build the variant after review.
    """
    from backend.task_engine import TaskSpec

    from .synthetic import SyntheticDataset, merge

    work = _work_dir(ds)
    if json.loads((work / "settings.json").read_text()).get("kind") == "test":
        raise ValueError("synthetic data is never added to an independent test set")
    dirs = [Path(d) for d in synthetic_dirs] if synthetic_dirs else sorted(work.glob("synthetic_*"))
    batches = [SyntheticDataset.model_validate_json((d / "synthetic.json").read_text())
               for d in dirs if (d / "synthetic.json").exists()]
    syn_frames = [f for b in batches for f in b.frames]
    if not syn_frames:
        raise ValueError(f"no kept synthetic images found in {[str(d) for d in dirs] or work}")

    task = TaskSpec.model_validate_json((work / "task_spec.json").read_text())
    frames = FrameCollection.model_validate_json((work / "frames.json").read_text())
    aset = get_annotations(ds)
    for b in batches:
        aset = merge(aset, b)
    settings = json.loads((work / "settings.json").read_text())
    if ds.split_strategy.startswith("by_session"):
        settings["val_sessions"] = ds.splits["val"].sessions if "val" in ds.splits else []

    dataset_id = new_id("ds")
    _, new_work, out = _dirs(dataset_id)
    new_work.mkdir(parents=True)
    merged_frames = frames.model_copy(update={"frames": frames.frames + syn_frames})
    (new_work / "task_spec.json").write_text(task.model_dump_json(indent=2))
    (new_work / "frames.json").write_text(merged_frames.model_dump_json(indent=2))
    (new_work / "annotations.json").write_text(aset.model_dump_json(indent=2))
    (new_work / "settings.json").write_text(json.dumps({**settings, "derived_from": ds.dataset_id,
                                                        "synthetic_dirs": [str(d) for d in dirs]}, indent=2))
    m = _export(task, aset, merged_frames, out, dataset_id, settings)
    m = m.model_copy(update={"warnings": m.warnings + [f"real+synthetic variant of {ds.dataset_id}"]})
    (out / "manifest.json").write_text(m.model_dump_json(indent=2))
    return m
