"""Standalone training CLI (no API, no other engines needed).

  python -m backend.training_engine.cli --data coco8.yaml --epochs 10
  python -m backend.training_engine.cli --data /data/datasets/vest/data.yaml --model yolo11n.pt --epochs 50 --imgsz 640
  python -m backend.training_engine.cli --data X.yaml --baseline-only     # pretrained model on val, no training

Prints one line "ZALU_RESULT {json}" (Ultralytics logs go to the other lines): split check, baseline metrics, fine-tuned metrics, timings, ONNX check, artifact paths.
Every run is also saved as a ModelArtifact, so it appears in GET /api/models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend import schemas
from backend.infrastructure import storage
from backend.training_engine import trainer
from backend.training_engine.splits import check_splits


def _resolve(data: str) -> tuple[str, list[str]]:
    """Accept a path or an Ultralytics dataset name (coco8.yaml); download sample datasets if needed."""
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(data)
    names = d["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    yaml_file = str(d.get("yaml_file") or data)
    return yaml_file, names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--variant", default="real_only", choices=["real_only", "real_plus_synthetic"])
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args(argv)

    yaml_file, names = _resolve(a.data)
    split = check_splits(yaml_file)
    out: dict = {"data_yaml": yaml_file, "classes": names[:10] + (["..."] if len(names) > 10 else []), "split_check": split}
    if not split["ok"]:
        out["error"] = "split check failed: refusing to train"
        print("ZALU_RESULT " + json.dumps(out), flush=True)
        return 2

    if a.baseline_only:
        out["baseline_metrics"] = trainer.baseline_val(a.model, yaml_file, names, a.imgsz)
        print("ZALU_RESULT " + json.dumps(out), flush=True)
        return 0

    ds = schemas.DatasetManifest(root=str(Path(yaml_file).parent), data_yaml=yaml_file, classes=names,
                                 variant=a.variant, split_check=split, counts=split["counts"],
                                 source_ids=[Path(a.data).stem], notes="registered by training CLI")
    storage.save("datasets", ds)
    req = schemas.TrainingRequest(dataset_id=ds.id, base_model=a.model, epochs=a.epochs, imgsz=a.imgsz,
                                  batch=a.batch, export_onnx=not a.no_onnx)

    def progress(p: float, msg: str) -> None:
        print(f"[{p:5.2f}] {msg}", file=sys.stderr, flush=True)

    art = trainer.train(req, ds, progress)
    storage.save("models", art)
    out.update({"dataset_id": ds.id, "model_id": art.id, "base_model": art.base_model, "epochs": art.epochs,
                "train_seconds": art.train_seconds, "baseline_metrics": art.baseline_metrics,
                "metrics": art.metrics, "weights": art.weights, "onnx": art.onnx,
                "export_check": art.export_check, "device": art.device})
    print("ZALU_RESULT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
