"""Build a (Kirill-format) DatasetManifest for a plain YOLO data.yaml: sample datasets, CLI runs, mocks.
The real Data Engine produces richer manifests (per-image list, sessions); this one carries counts only."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend import schemas
from backend.training_engine.splits import check_splits, resolve


def manifest_from_yaml(data_yaml: str, task_id: Optional[str] = None, variant: str = "real_only",
                       note: Optional[str] = None) -> tuple[schemas.DatasetManifest, dict]:
    cfg = resolve(data_yaml)
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    check = check_splits(data_yaml)
    m = schemas.DatasetManifest(
        task_id=task_id or "task_none",
        variant=variant,
        root=str(cfg.get("path") or Path(data_yaml).parent),
        data_yaml=str(cfg.get("yaml_file") or data_yaml),
        class_names=names,
        splits={k: schemas.SplitStats(images=n) for k, n in check["counts"].items() if n},
        split_strategy="provided data.yaml (not session-split by the Data Engine)",
        images=[],
        warnings=([note] if note else []) + check["warnings"] + check["errors"],
    )
    return m, check
