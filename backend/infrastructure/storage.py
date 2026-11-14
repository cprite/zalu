"""File-based storage: one JSON file per object. Good enough for a hackathon, easy to inspect."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

DATA_ROOT = Path(os.environ.get("ZALU_DATA", "/data")).resolve()
KINDS = {"tasks", "datasets", "jobs", "models", "reports"}
_lock = threading.Lock()

T = TypeVar("T", bound=BaseModel)


def path(*parts: str) -> Path:
    p = DATA_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(kind: str, obj: BaseModel) -> None:
    assert kind in KINDS, kind
    from backend.schemas import obj_id

    p = path(kind, f"{obj_id(obj)}.json")
    tmp = p.with_suffix(".tmp")
    with _lock:
        tmp.write_text(obj.model_dump_json(indent=1))
        os.replace(tmp, p)


def load(kind: str, obj_id: str, model: Type[T]) -> Optional[T]:
    assert kind in KINDS, kind
    if "/" in obj_id or ".." in obj_id:
        return None
    p = DATA_ROOT / kind / f"{obj_id}.json"
    if not p.exists():
        return None
    return model.model_validate_json(p.read_text())


def list_all(kind: str, model: Type[T]) -> list[T]:
    d = DATA_ROOT / kind
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.json"):
        try:
            out.append(model.model_validate_json(p.read_text()))
        except Exception:  # files written by an older schema version are skipped, not fatal
            continue
    return sorted(out, key=lambda o: str(getattr(o, "created_at", "")), reverse=True)


def write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
