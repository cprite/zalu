"""process_media(files) -> FrameCollection.

Extracts frames from videos (OpenCV), loads images, drops unreadable / blurry /
near-duplicate frames, then picks a diverse subset per recording session.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .schema import Frame, FrameCollection, RejectedFrame, new_id

log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PathLike = Union[str, Path]


# ---------------------------------------------------------------- image metrics
def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    if w > 640:
        gray = cv2.resize(gray, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _feature(img: np.ndarray) -> np.ndarray:
    """Cheap appearance descriptor for diversity sampling."""
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (24, 24), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32).flatten()
    g = (g - g.mean()) / (g.std() + 1e-6)
    hsv = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6) * 10.0
    return np.concatenate([g / np.sqrt(len(g)), hist])


def farthest_point_sample(feats: np.ndarray, k: int) -> List[int]:
    """Greedy max-min selection, seeded with the first frame (deterministic)."""
    n = len(feats)
    if k >= n:
        return list(range(n))
    chosen = [0]
    dist = np.linalg.norm(feats - feats[0], axis=1)
    while len(chosen) < k:
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[i], axis=1))
    return sorted(chosen)


# ---------------------------------------------------------------- readers
def _iter_video(path: Path, fps: float) -> Iterator[Tuple[int, float, Optional[np.ndarray]]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        yield -1, 0.0, None
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0
    step = max(1, int(round(src_fps / fps)))
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, img = cap.read()
                if not ok:
                    break
                yield idx, idx / src_fps, img
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)


def default_session_id(path: Path) -> str:
    """One video = one session; loose images are grouped by their folder."""
    key = path if path.suffix.lower() in VIDEO_EXT else path.parent
    h = hashlib.sha1(str(key.resolve()).encode()).hexdigest()[:6]
    stem = "".join(c if c.isalnum() else "_" for c in key.stem or key.name)[:24]
    return f"s_{stem}_{h}"


def _expand(files: Iterable[PathLike]) -> List[Path]:
    out: List[Path] = []
    for f in files:
        p = Path(f)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXT | IMAGE_EXT)
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------- main entry
def process_media(
    files: Sequence[PathLike],
    out_dir: PathLike,
    task=None,  # TaskSpec, unused for now; reserved for task-aware frame selection
    session_ids: Optional[Dict[str, str]] = None,
    fps: float = 1.0,
    max_frames: Optional[int] = 200,
    min_sharpness: float = 25.0,
    dedup_distance: int = 5,
    min_side: int = 160,
    max_side: int = 1280,
) -> FrameCollection:
    """Turn uploaded videos/images into a cleaned, diverse FrameCollection.

    session_ids: optional {file path: session id}. Files of the same physical
    recording should share an id; different recordings must not.
    """
    out_dir = Path(out_dir).resolve()
    frames_dir = out_dir / "frames"
    session_ids = {str(Path(k).resolve()): v for k, v in (session_ids or {}).items()}

    candidates: List[Tuple[Frame, np.ndarray]] = []
    rejected: List[RejectedFrame] = []
    sessions: Dict[str, List[str]] = {}
    kept_hashes: List[int] = []

    for path in _expand(files):
        ext = path.suffix.lower()
        sid = session_ids.get(str(path.resolve())) or default_session_id(path)
        sessions.setdefault(sid, []).append(str(path))
        if ext in VIDEO_EXT:
            source = _iter_video(path, fps)
        elif ext in IMAGE_EXT:
            source = iter([(None, None, cv2.imread(str(path)))])
        else:
            rejected.append(RejectedFrame(source_file=str(path), reason="unreadable", detail=f"unsupported type {ext}"))
            continue

        n_read = 0
        for idx, ts, img in source:
            n_read += 1
            if img is None:
                rejected.append(RejectedFrame(source_file=str(path), source_frame_index=idx, reason="unreadable"))
                continue
            if min(img.shape[:2]) < min_side:
                rejected.append(RejectedFrame(source_file=str(path), source_frame_index=idx, reason="too_small",
                                              detail=f"{img.shape[1]}x{img.shape[0]}"))
                continue
            img = _resize_max(img, max_side)
            sharp = sharpness(img)
            if sharp < min_sharpness:
                rejected.append(RejectedFrame(source_file=str(path), source_frame_index=idx, reason="blurry",
                                              detail=f"sharpness {sharp:.1f} < {min_sharpness}"))
                continue
            h = dhash(img)
            near = next((k for k in kept_hashes if hamming(h, k) <= dedup_distance), None)
            if near is not None:
                rejected.append(RejectedFrame(source_file=str(path), source_frame_index=idx, reason="near_duplicate",
                                              detail=f"hamming {hamming(h, near)} <= {dedup_distance}"))
                continue
            kept_hashes.append(h)
            fid = new_id("f")
            candidates.append((Frame(
                frame_id=fid, path=str(frames_dir / sid / f"{fid}.jpg"), session_id=sid, source_file=str(path),
                source_frame_index=idx, timestamp_s=None if ts is None else round(ts, 3),
                width=img.shape[1], height=img.shape[0], sharpness=round(sharp, 1), dhash=f"{h:016x}",
            ), img))
        if n_read == 0:
            rejected.append(RejectedFrame(source_file=str(path), reason="unreadable", detail="no frames decoded"))

    selected = _select_diverse(candidates, max_frames, rejected)

    frames: List[Frame] = []
    for fr, img in selected:
        Path(fr.path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(fr.path, img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError(f"failed to write {fr.path}")
        frames.append(fr)

    coll = FrameCollection(
        root=str(out_dir), frames=frames, rejected=rejected, sessions=sessions,
        settings=dict(fps=fps, max_frames=max_frames, min_sharpness=min_sharpness,
                      dedup_distance=dedup_distance, min_side=min_side, max_side=max_side),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames.json").write_text(coll.model_dump_json(indent=2))
    log.info("process_media: %d frames kept, %d rejected, %d sessions", len(frames), len(rejected), len(sessions))
    return coll


def _select_diverse(cands, max_frames, rejected):
    if not max_frames or len(cands) <= max_frames:
        return cands
    by_session: Dict[str, list] = {}
    for c in cands:
        by_session.setdefault(c[0].session_id, []).append(c)
    total = len(cands)
    # Proportional quota, at least 1 frame per session.
    quota = {s: max(1, int(round(max_frames * len(v) / total))) for s, v in by_session.items()}
    out = []
    for s, items in by_session.items():
        feats = np.stack([_feature(img) for _, img in items])
        keep = set(farthest_point_sample(feats, quota[s]))
        for i, (fr, img) in enumerate(items):
            if i in keep:
                out.append((fr, img))
            else:
                rejected.append(RejectedFrame(source_file=fr.source_file, source_frame_index=fr.source_frame_index,
                                              reason="not_selected", detail="diversity sampling"))
    return out
