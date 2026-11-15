"""File-based storage for video search, under ZALU_DATA/video_search/ (same JSON-file pattern as
backend/infrastructure/storage.py; atomic writes via tmp + os.replace).

Layout:
  video_search/batches/{batch_id}/batch.json            batch + per-recording state (also the job ledger)
  video_search/batches/{batch_id}/media/{rec_id}{ext}   uploaded source recording
  video_search/batches/{batch_id}/tracklets/{rec_id}.json    tracklets of one recording
  video_search/batches/{batch_id}/embeddings/{rec_id}.npy    L2-normalised float32 [n, d], row i = ids[i]
  video_search/batches/{batch_id}/embeddings/{rec_id}.json   {"version", "ids"}
  video_search/thumbs/{tracklet_id}.jpg                 representative crop
  video_search/tracklet_index/{tracklet_id}.json        {"batch_id"} (tracklet -> batch lookup)
  video_search/searches/{search_id}.json                stored search result
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from backend.infrastructure import storage
from backend.video_search.schemas import Batch, Tracklet

_ID = re.compile(r"^[a-z]+_[0-9a-f]{6,32}$")
_lock = threading.RLock()


def valid_id(x: str) -> bool:
    return bool(_ID.match(x or ""))


def root() -> Path:
    return storage.DATA_ROOT / "video_search"


def batch_dir(batch_id: str) -> Path:
    assert valid_id(batch_id), batch_id
    return root() / "batches" / batch_id


def _atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def save_batch(b: Batch) -> None:
    with _lock:
        _atomic_write(batch_dir(b.id) / "batch.json", b.model_dump_json(indent=1))


def load_batch(batch_id: str) -> Optional[Batch]:
    if not valid_id(batch_id):
        return None
    p = batch_dir(batch_id) / "batch.json"
    return Batch.model_validate_json(p.read_text()) if p.exists() else None


def list_batches() -> list[Batch]:
    d = root() / "batches"
    out = []
    for p in sorted(d.glob("*/batch.json")) if d.exists() else []:
        try:
            out.append(Batch.model_validate_json(p.read_text()))
        except Exception:
            continue
    return out


def update_recording(batch_id: str, rec_id: str, **fields) -> Batch:
    """Read-modify-write of one recording under the lock (the worker thread and the API both write)."""
    with _lock:
        b = load_batch(batch_id)
        for r in b.recordings:
            if r.id == rec_id:
                for k, v in fields.items():
                    setattr(r, k, v)
        save_batch(b)
        return b


def media_path(batch_id: str, rec_id: str, filename: str) -> Path:
    assert valid_id(rec_id), rec_id
    ext = Path(filename).suffix.lower()
    ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext or "") else ".bin"
    return batch_dir(batch_id) / "media" / f"{rec_id}{ext}"


def save_tracklets(batch_id: str, rec_id: str, tracklets: list[Tracklet]) -> None:
    """Per-recording file, so a retried recording replaces only its own tracklets."""
    with _lock:
        _atomic_write(batch_dir(batch_id) / "tracklets" / f"{rec_id}.json",
                      json.dumps([t.model_dump() for t in tracklets], indent=1))
        for t in tracklets:
            _atomic_write(root() / "tracklet_index" / f"{t.tracklet_id}.json", json.dumps({"batch_id": batch_id}))


def load_tracklets(batch_id: str) -> list[Tracklet]:
    d = batch_dir(batch_id) / "tracklets"
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        out += [Tracklet(**x) for x in json.loads(p.read_text())]
    return out


def get_tracklet(tracklet_id: str) -> Optional[Tracklet]:
    bid = tracklet_batch(tracklet_id)
    if not bid:
        return None
    return next((t for t in load_tracklets(bid) if t.tracklet_id == tracklet_id), None)


def tracklet_batch(tracklet_id: str) -> Optional[str]:
    if not valid_id(tracklet_id):
        return None
    p = root() / "tracklet_index" / f"{tracklet_id}.json"
    return json.loads(p.read_text())["batch_id"] if p.exists() else None


def thumb_path(tracklet_id: str) -> Path:
    assert valid_id(tracklet_id), tracklet_id
    p = root() / "thumbs" / f"{tracklet_id}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_embeddings(batch_id: str, rec_id: str, ids: list[str], vecs: np.ndarray, version: str) -> None:
    d = batch_dir(batch_id) / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    with _lock:
        tmp = d / f"{rec_id}.tmp.npy"
        np.save(tmp, vecs.astype(np.float32))
        os.replace(tmp, d / f"{rec_id}.npy")
        _atomic_write(d / f"{rec_id}.json", json.dumps({"version": version, "ids": ids}))


def load_embeddings(batch_id: str, version: str) -> tuple[list[str], Optional[np.ndarray]]:
    """All rows of the batch index with the given embedding_version (other versions are ignored, not mixed)."""
    d = batch_dir(batch_id) / "embeddings"
    ids, mats = [], []
    for meta_p in sorted(d.glob("*.json")) if d.exists() else []:
        meta = json.loads(meta_p.read_text())
        if meta["version"] != version or not meta["ids"]:
            continue
        ids += meta["ids"]
        mats.append(np.load(meta_p.with_suffix(".npy")))
    return ids, (np.concatenate(mats) if mats else None)


def save_search(search: dict) -> None:
    _atomic_write(root() / "searches" / f"{search['search_id']}.json", json.dumps(search, indent=1))


def load_search(search_id: str) -> Optional[dict]:
    if not valid_id(search_id):
        return None
    p = root() / "searches" / f"{search_id}.json"
    return json.loads(p.read_text()) if p.exists() else None
