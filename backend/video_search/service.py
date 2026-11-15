"""Video search service: GX10 pipeline adapter, greedy IOU tracker, background batch processing, CLIP search.

Split (same as the rest of ZALU): nikopi5 keeps the uploaded recording, metadata, thumbnails and the embedding
index; the GX10 worker runs person detection (/video_infer, zero-shot YOLOE prompted with "person") and CLIP
(/clip/image, /clip/text, added to backend/gpu_worker.py in this change).
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from backend import schemas as core
from backend.infrastructure import gpu_client, storage
from backend.video_search import repository as repo
from backend.video_search.schemas import EMBEDDING_VERSION, SCORE_TYPE, Batch, Recording, SearchRequest, Tracklet

log = logging.getLogger("zalu.video_search")

IOU_MATCH = 0.3        # min IOU to continue a track
MAX_MISSED = 2         # sampled frames a track may miss before it is closed (2 at 2 fps = 1 s)
CROP_PAD = 0.10        # representative crop padding, fraction of box size
TEXT_TEMPLATE = "a photo of {}"


# ---------- media probe ----------
def ffprobe(path: Path) -> dict:
    """duration_s / fps / width / height / codec via ffprobe; every field null if ffprobe is missing or fails."""
    out = {"duration_s": None, "fps": None, "width": None, "height": None, "codec": None, "probe_error": None}
    exe = shutil.which("ffprobe")
    if not exe:
        out["probe_error"] = "ffprobe not installed"
        return out
    try:
        r = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                            "-select_streams", "v:0", str(path)], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            out["probe_error"] = r.stderr.strip()[:300] or f"ffprobe exit {r.returncode}"
            return out
        j = json.loads(r.stdout)
        st = (j.get("streams") or [{}])[0]
        dur = (j.get("format") or {}).get("duration") or st.get("duration")
        out["duration_s"] = round(float(dur), 3) if dur else None
        rate = st.get("avg_frame_rate") or st.get("r_frame_rate")
        if rate and rate != "0/0":
            out["fps"] = round(float(Fraction(rate)), 3)
        out["width"], out["height"], out["codec"] = st.get("width"), st.get("height"), st.get("codec_name")
        if not st:
            out["probe_error"] = "no video stream"
    except Exception as e:  # never fail an upload because of probing
        out["probe_error"] = f"{type(e).__name__}: {e}"[:300]
    return out


# ---------- GX10 pipeline (replaceable; tests inject a fake) ----------
class Pipeline(Protocol):
    def upload(self, batch_id: str, local: Path) -> str: ...
    def detect(self, model: core.ModelArtifact, gx10_path: str, fps: float, conf: float) -> dict: ...
    def embed_images(self, jpegs: list[bytes]) -> np.ndarray: ...
    def embed_text(self, texts: list[str]) -> np.ndarray: ...


class GX10Pipeline:
    def upload(self, batch_id: str, local: Path) -> str:
        import requests

        url = gpu_client.GPU_URL + "/upload"
        with local.open("rb") as fh:
            try:  # streamed multipart body when requests_toolbelt is available (no full copy in memory here)
                from requests_toolbelt import MultipartEncoder

                enc = MultipartEncoder({"subdir": f"video_search/{batch_id}",
                                        "files": (local.name, fh, "application/octet-stream")})
                r = requests.post(url, data=enc, headers={"Content-Type": enc.content_type}, timeout=1800)
            except ImportError:
                r = requests.post(url, data={"subdir": f"video_search/{batch_id}"},
                                  files=[("files", (local.name, fh, "application/octet-stream"))], timeout=1800)
        return gpu_client._check(r)["paths"][0]

    def detect(self, model, gx10_path, fps, conf):
        return gpu_client.post("/video_infer", json={"model": model.model_dump(), "path": gx10_path,
                                                     "fps": fps, "conf": conf}, timeout=3600)

    def embed_images(self, jpegs):
        files = [("images", (f"{i}.jpg", b, "image/jpeg")) for i, b in enumerate(jpegs)]
        return np.asarray(gpu_client.post("/clip/image", files=files, timeout=600)["embeddings"], dtype=np.float32)

    def embed_text(self, texts):
        return np.asarray(gpu_client.post("/clip/text", json={"texts": texts}, timeout=120)["embeddings"],
                          dtype=np.float32)


PIPELINE: Pipeline = GX10Pipeline()


# ---------- tracking ----------
def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_tracks(frames: list[dict], cls: str = "person", iou_min: float = IOU_MATCH,
                 max_missed: int = MAX_MISSED) -> list[list[dict]]:
    """Greedy IOU tracker over sampled frames. frames: [{frame_index, t_s, detections:[{cls, score, box}]}].
    Returns a list of tracks; each track is a list of {frame_index, t_s, score, box} in time order."""
    open_tracks: list[dict] = []  # {"obs": [...], "missed": int}
    done: list[list[dict]] = []
    for fr in sorted(frames, key=lambda f: f["frame_index"]):
        dets = [d for d in fr.get("detections", []) if d.get("cls") == cls]
        pairs = sorted(((iou(t["obs"][-1]["box"], d["box"]), ti, di) for ti, t in enumerate(open_tracks)
                        for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for s, ti, di in pairs:
            if s < iou_min:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti), used_d.add(di)
            d = dets[di]
            open_tracks[ti]["obs"].append({"frame_index": fr["frame_index"], "t_s": fr["t_s"],
                                           "score": d["score"], "box": d["box"]})
            open_tracks[ti]["missed"] = 0
        still = []
        for ti, t in enumerate(open_tracks):
            if ti not in used_t:
                t["missed"] += 1
            (done.append(t["obs"]) if t["missed"] > max_missed else still.append(t))
        open_tracks = still
        for di, d in enumerate(dets):
            if di not in used_d:
                open_tracks.append({"obs": [{"frame_index": fr["frame_index"], "t_s": fr["t_s"],
                                             "score": d["score"], "box": d["box"]}], "missed": 0})
    done += [t["obs"] for t in open_tracks]
    return sorted(done, key=lambda o: o[0]["frame_index"])


def _rep(obs: list[dict]) -> dict:
    """Representative observation: highest score x sqrt(area) (confident and not tiny)."""
    return max(obs, key=lambda o: o["score"] * max(1.0, (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])) ** 0.5)


def crop_jpegs(video: Path, reps: list[dict]) -> list[Optional[bytes]]:
    """Decode the representative frames from the local copy and crop the padded box (JPEG bytes, None on failure)."""
    import cv2

    out: list[Optional[bytes]] = [None] * len(reps)
    cap = cv2.VideoCapture(str(video))
    try:
        for i in sorted(range(len(reps)), key=lambda k: reps[k]["frame_index"]):
            o = reps[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, o["frame_index"])
            ok, f = cap.read()
            if not ok:
                continue
            h, w = f.shape[:2]
            x1, y1, x2, y2 = o["box"]
            px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
            x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            ok, buf = cv2.imencode(".jpg", f[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])
            out[i] = buf.tobytes() if ok else None
    finally:
        cap.release()
    return out


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(np.asarray(m, dtype=np.float32))
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


# ---------- processing ----------
_q: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def ensure_worker() -> None:
    """Start the single background worker; on first start, requeue what a restart interrupted (on-disk ledger)."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
        for b in repo.list_batches():
            stuck = [r for r in b.recordings if r.state == "processing"]
            for r in stuck:
                repo.update_recording(b.id, r.id, state="queued", stage=None,
                                      error="requeued: server restarted while processing")
            if b.status in ("queued", "processing") or stuck:
                _q.put(b.id)
        threading.Thread(target=_loop, daemon=True, name="video-search-worker").start()


def _loop() -> None:
    while True:
        bid = _q.get()
        try:
            process_batch(bid)
        except Exception:  # never kill the worker thread
            log.exception("video search batch %s crashed", bid)


def enqueue(batch: Batch) -> Batch:
    with repo._lock:
        b = repo.load_batch(batch.id)
        for r in b.recordings:
            if r.state in ("uploaded", "failed"):
                r.state, r.stage, r.error = "queued", None, None
        b.status, b.queued_at, b.finished_at = "queued", core.now(), None
        repo.save_batch(b)
    ensure_worker()
    _q.put(b.id)
    return b


def _person_model(b: Batch) -> core.ModelArtifact:
    """One zero-shot person artifact per batch, reused for every recording (same as POST /api/models/zero-shot)."""
    if b.model_id:
        art = storage.load("models", b.model_id, core.ModelArtifact)
        if art:
            return art
    from backend.training_engine.trainer import zero_shot_artifact

    art = zero_shot_artifact(["person"])
    storage.save("models", art)
    with repo._lock:
        cur = repo.load_batch(b.id)
        cur.model_id = art.id
        repo.save_batch(cur)
    return art


def process_recording(b: Batch, r: Recording, model: core.ModelArtifact) -> None:
    t_all = time.time()
    timings: dict = {}
    repo.update_recording(b.id, r.id, state="processing", stage="gx10_upload", started_at=core.now(), error=None)
    local = repo.media_path(b.id, r.id, r.filename)
    t0 = time.time()
    gx10_path = r.gx10_path or PIPELINE.upload(b.id, local)
    timings["gx10_upload"] = round(time.time() - t0, 2)
    repo.update_recording(b.id, r.id, stage="detection", gx10_path=gx10_path, timings_s=dict(timings))

    t0 = time.time()
    res = PIPELINE.detect(model, gx10_path, b.fps, b.conf)
    timings["detection"] = round(time.time() - t0, 2)
    frames = res.get("frames", [])
    n_det = sum(1 for f in frames for d in f.get("detections", []) if d.get("cls") == "person")
    repo.update_recording(b.id, r.id, stage="tracking", sampled_frames=len(frames), detections=n_det,
                          frames_with_person=sum(1 for f in frames if any(d.get("cls") == "person" for d in f.get("detections", []))),
                          timings_s=dict(timings))

    t0 = time.time()
    tracks = build_tracks(frames)
    reps = [_rep(o) for o in tracks]
    tracklets = []
    for obs, rep in zip(tracks, reps):
        tracklets.append(Tracklet(
            tracklet_id=core.new_id("trk"), batch_id=b.id, recording_id=r.id, camera_id=r.camera_id,
            start_ms=int(round(obs[0]["t_s"] * 1000)), end_ms=int(round(obs[-1]["t_s"] * 1000)),
            detection_count=len(obs), max_score=max(o["score"] for o in obs),
            rep_frame_index=rep["frame_index"], rep_t_ms=int(round(rep["t_s"] * 1000)), rep_box=rep["box"]))
    timings["tracking"] = round(time.time() - t0, 2)
    repo.update_recording(b.id, r.id, stage="embedding", timings_s=dict(timings))

    t0 = time.time()
    jpegs = crop_jpegs(local, reps)
    keep = [i for i, j in enumerate(jpegs) if j]
    for i in keep:
        repo.thumb_path(tracklets[i].tracklet_id).write_bytes(jpegs[i])
    vecs = PIPELINE.embed_images([jpegs[i] for i in keep]) if keep else np.zeros((0, 1), np.float32)
    if len(vecs) != len(keep):
        raise RuntimeError(f"CLIP returned {len(vecs)} embeddings for {len(keep)} crops")
    for i in keep:
        tracklets[i].embedding_version = EMBEDDING_VERSION
    repo.save_tracklets(b.id, r.id, tracklets)
    repo.save_embeddings(b.id, r.id, [tracklets[i].tracklet_id for i in keep], _norm(vecs) if keep else vecs,
                         EMBEDDING_VERSION)
    timings["embedding"] = round(time.time() - t0, 2)
    timings["total"] = round(time.time() - t_all, 2)
    repo.update_recording(b.id, r.id, state="completed", stage="done", tracklets=len(tracklets),
                          finished_at=core.now(), timings_s=timings,
                          error=None if len(keep) == len(tracklets) else
                          f"{len(tracklets) - len(keep)} tracklet crops could not be decoded (not searchable)")


def process_batch(batch_id: str) -> Batch:
    b = repo.load_batch(batch_id)
    if b is None:
        return None
    with repo._lock:
        cur = repo.load_batch(batch_id)
        cur.status = "processing"
        repo.save_batch(cur)
    model = None
    for r in b.recordings:
        r = next(x for x in repo.load_batch(batch_id).recordings if x.id == r.id)
        if r.state != "queued":
            continue
        try:
            model = model or _person_model(b)
            process_recording(b, r, model)
        except Exception as e:
            log.exception("recording %s failed", r.id)
            msg = e.detail if isinstance(e, gpu_client.GPUWorkerError) else f"{type(e).__name__}: {e}"
            repo.update_recording(b.id, r.id, state="failed", error=str(msg)[:1000], finished_at=core.now())
    with repo._lock:
        cur = repo.load_batch(batch_id)
        states = [x.state for x in cur.recordings if x.state != "created"]  # never-uploaded ones are skipped
        if any(s == "queued" for s in states):  # retried while we ran: the queued entry picks it up
            cur.status = "queued"
        else:
            ok = sum(s == "completed" for s in states)
            cur.status = "completed" if ok == len(states) else ("failed" if ok == 0 else "partial")
            cur.finished_at = core.now()
        repo.save_batch(cur)
        return cur


# ---------- progress ----------
def progress(b: Batch) -> dict:
    counts = {s: 0 for s in ("created", "uploaded", "queued", "processing", "completed", "failed")}
    for r in b.recordings:
        counts[r.state] += 1
    total = len(b.recordings)
    recs = [r.model_dump(exclude={"gx10_path"}, mode="json") for r in b.recordings]
    return {"batch_id": b.id, "name": b.name, "status": b.status, "fps": b.fps, "conf": b.conf,
            "total_recordings": total, **{f"{k}_recordings": v for k, v in counts.items()},
            "progress": {"finished_recordings": counts["completed"] + counts["failed"], "total_recordings": total,
                         "fraction": round((counts["completed"] + counts["failed"]) / total, 3) if total else 0.0,
                         "note": "measured per recording stage; /video_infer is one blocking GX10 call, so there "
                                 "is no frame-level progress inside the detection stage"},
            "created_at": b.created_at, "queued_at": b.queued_at, "finished_at": b.finished_at,
            "sampling": {"fps": b.fps, "timing_uncertainty_ms": int(round(1000 / b.fps)),
                         "time_base": "frame_index / container fps (not PTS)"},
            "recordings": recs}


# ---------- search ----------
def _utc(start, ms: int) -> Optional[str]:
    if start is None:
        return None
    return (start + timedelta(milliseconds=ms)).isoformat().replace("+00:00", "Z")


def search(b: Batch, req: SearchRequest) -> dict:
    """Rank the batch's stored tracklet embeddings by cosine similarity with the CLIP text embedding of the query.
    Uses only the stored index: no detection, no image encoding."""
    t0 = time.time()
    ids, mat = repo.load_embeddings(b.id, EMBEDDING_VERSION)
    sid = core.new_id("search")
    base = {"search_id": sid, "batch_id": b.id, "query": req.query, "camera_ids": req.camera_ids,
            "top_k": req.top_k, "min_score": req.min_score, "score_type": SCORE_TYPE,
            "embedding_version": EMBEDDING_VERSION, "created_at": core.now()}
    if mat is None or not ids:
        out = {**base, "status": "completed", "total_matches": 0, "indexed_tracklets": 0, "results": [],
               "note": "no indexed tracklets yet (batch not processed, or no person detected)"}
        repo.save_search(out)
        return out
    q = _norm(PIPELINE.embed_text([TEXT_TEMPLATE.format(req.query.strip())]))[0]
    scores = mat @ q
    trk = {t.tracklet_id: t for t in repo.load_tracklets(b.id)}
    recs = {r.id: r for r in b.recordings}
    cams = set(req.camera_ids)
    results = []
    for i in np.argsort(-scores):
        t = trk.get(ids[i])
        if t is None or (cams and t.camera_id not in cams):
            continue
        s = float(scores[i])
        if req.min_score is not None and s < req.min_score:
            break
        start = recs[t.recording_id].recording_start_utc if t.recording_id in recs else None
        results.append({
            "tracklet_id": t.tracklet_id, "camera_id": t.camera_id, "recording_id": t.recording_id,
            "start_ms": t.start_ms, "end_ms": t.end_ms, "visible_duration_ms": t.end_ms - t.start_ms,
            "start_utc": _utc(start, t.start_ms), "end_utc": _utc(start, t.end_ms),
            "query_score": round(s, 4), "score_type": SCORE_TYPE, "detection_count": t.detection_count,
            "cross_camera_identity_status": "not_assessed",
            "thumbnail_url": f"/api/video-search/tracklets/{t.tracklet_id}/thumbnail",
            "playback_url": f"/api/video-search/recordings/{t.recording_id}/playback?start_ms={t.start_ms}"})
        if len(results) >= req.top_k:
            break
    out = {**base, "status": "completed", "total_matches": len(results), "indexed_tracklets": len(ids),
           "search_ms": round((time.time() - t0) * 1000, 1), "results": results,
           "note": "query_score is raw CLIP cosine similarity (uncalibrated; compare within one query only)"}
    repo.save_search(out)
    return out
