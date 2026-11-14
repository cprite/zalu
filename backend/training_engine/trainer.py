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
        dataset_id=dataset.dataset_id,
        variant=dataset.variant,
        classes=dataset.class_names,
        epochs=req.epochs,
        imgsz=req.imgsz,
        device=device,
    )
    out_dir = storage.path("runs", art.id, "x").parent

    if req.baseline:
        if progress:
            progress(0.01, "baseline: pretrained model on val split")
        art.baseline_metrics = baseline_val(req.base_model, dataset.data_yaml, dataset.class_names, req.imgsz)

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
        if req.verify_export:
            art.export_check = verify_onnx(str(weights), art.onnx, dataset.data_yaml, req.imgsz)

    return art


def _val_image(data_yaml: str) -> Optional[str]:
    from backend.training_engine.splits import _images, resolve

    cfg = resolve(data_yaml)
    root = Path(cfg.get("path") or Path(data_yaml).parent)
    imgs = _images(root, cfg.get("val"))
    return str(imgs[0]) if imgs else None


def baseline_val(base_model: str, data_yaml: str, classes: list[str], imgsz: int = 640) -> dict[str, float]:
    """Pretrained model, no fine-tuning, on the same val split. Only meaningful when the dataset's class names
    are a subset of the base model's classes (e.g. COCO 'person'); otherwise returns {} and the open-vocabulary
    zero-shot artifact is the baseline instead."""
    from ultralytics import YOLO

    m = YOLO(base_model)
    names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
    if not set(classes) <= set(names) or list(names[: len(classes)]) != list(classes):
        return {}
    r = m.val(data=data_yaml, imgsz=imgsz, device=0, plots=False, verbose=False, split="val")
    return {"precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
            "map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4)}


def verify_onnx(pt: str, onnx: str, data_yaml: str, imgsz: int = 640) -> dict:
    """Run the exported ONNX and the .pt on one val image; they must find the same number of objects."""
    import time as _t

    from ultralytics import YOLO

    img = _val_image(data_yaml)
    if img is None:
        return {"ok": False, "error": "no val image to test with"}
    try:
        a = YOLO(pt).predict(img, imgsz=imgsz, conf=0.25, device=0, verbose=False)[0]
        mo = YOLO(onnx, task="detect")
        mo.predict(img, imgsz=imgsz, conf=0.25, verbose=False)  # warm-up
        t0 = _t.perf_counter()
        b = mo.predict(img, imgsz=imgsz, conf=0.25, verbose=False)[0]
        ms = (_t.perf_counter() - t0) * 1000
        na, nb = len(a.boxes), len(b.boxes)
        return {"ok": abs(na - nb) <= 1, "pt_detections": na, "onnx_detections": nb,
                "onnx_cpu_ms": round(ms, 1), "image": Path(img).name}
    except Exception as e:  # report, don't fail the whole training job
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def zero_shot_artifact(classes: list[str], base_model: str = "yoloe-11l-seg.pt") -> schemas.ModelArtifact:
    """The pretrained baseline: an open-vocabulary model prompted with the class names, no training."""
    return schemas.ModelArtifact(kind="zero_shot", base_model=base_model, classes=classes)
