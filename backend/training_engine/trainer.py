"""Training Engine: fine-tune a compact YOLO model on a YOLO-format dataset, save metrics, export ONNX.

Runs in its own process (see run_job.py) so a crash or CUDA error never takes the API down.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Optional

from backend import schemas
from backend.infrastructure import storage


def _last_metrics(results_csv: Path) -> dict[str, float]:
    """Read the final epoch's validation metrics from Ultralytics' results.csv."""
    if not results_csv.exists():
        return {}
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        return {}
    last = {k.strip(): v for k, v in rows[-1].items()}
    pick = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    out = {}
    for k, col in pick.items():
        try:
            out[k] = round(float(last[col]), 4)
        except (KeyError, ValueError):
            pass
    return out


def train(
    req: schemas.TrainingRequest,
    dataset: schemas.DatasetManifest,
    progress: Optional[Callable[[float, str], None]] = None,
) -> schemas.ModelArtifact:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: refusing to train on CPU (check --gpus all)")
    device = torch.cuda.get_device_name(0)
    art = schemas.ModelArtifact(
        base_model=req.base_model,
        dataset_id=dataset.id,
        variant=dataset.variant,
        classes=dataset.classes,
        epochs=req.epochs,
        imgsz=req.imgsz,
        device=device,
    )
    out_dir = storage.path("runs", art.id, "x").parent

    model = YOLO(req.base_model)

    def on_epoch_end(trainer):  # Ultralytics callback
        if progress:
            ep = trainer.epoch + 1
            progress(min(0.95, ep / req.epochs * 0.95), f"epoch {ep}/{req.epochs}")

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    t0 = time.time()
    model.train(
        data=dataset.data_yaml,
        epochs=req.epochs,
        imgsz=req.imgsz,
        batch=req.batch,
        device=0,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        plots=False,
        verbose=False,
        workers=4,
    )
    art.train_seconds = round(time.time() - t0, 1)

    weights = out_dir / "weights" / "best.pt"
    if not weights.exists():
        raise RuntimeError(f"training finished but {weights} is missing")
    art.weights = str(weights)
    art.metrics = _last_metrics(out_dir / "results.csv")

    if req.export_onnx:
        if progress:
            progress(0.97, "exporting ONNX")
        onnx_path = YOLO(str(weights)).export(format="onnx", imgsz=req.imgsz)
        art.onnx = str(onnx_path)

    return art


def zero_shot_artifact(classes: list[str], base_model: str = "yoloe-11l-seg.pt") -> schemas.ModelArtifact:
    """The pretrained baseline: an open-vocabulary model prompted with the class names, no training."""
    return schemas.ModelArtifact(kind="zero_shot", base_model=base_model, classes=classes)
