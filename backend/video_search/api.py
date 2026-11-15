"""FastAPI router for video search, mounted by backend/main.py under /api/video-search with the same bearer auth."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.video_search import repository as repo
from backend.video_search import service
from backend.video_search.schemas import Batch, BatchCreate, Recording, SearchRequest

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.environ.get("ZALU_VS_MAX_UPLOAD_BYTES", str(4 * 1024 ** 3)))  # 4 GiB per recording
CHUNK = 1024 * 1024


def _batch(batch_id: str) -> Batch:
    b = repo.load_batch(batch_id)
    if b is None:
        raise HTTPException(404, f"batch {batch_id} not found")
    return b


def _rec(b: Batch, rec_id: str) -> Recording:
    r = next((x for x in b.recordings if x.id == rec_id), None)
    if r is None:
        raise HTTPException(404, f"recording {rec_id} not found in batch {b.id}")
    return r


@router.post("/batches")
def create_batch(body: BatchCreate):
    b = Batch(name=body.name, fps=body.fps, conf=body.conf,
              recordings=[Recording(**x.model_dump()) for x in body.recordings])
    repo.save_batch(b)
    return {"batch_id": b.id, "status": b.status,
            "recordings": [{"recording_id": r.id, "camera_id": r.camera_id, "filename": r.filename,
                            "upload_url": f"/api/video-search/batches/{b.id}/recordings/{r.id}/upload"}
                           for r in b.recordings]}


@router.get("/batches")
def list_batches():
    return [{"batch_id": b.id, "name": b.name, "status": b.status, "total_recordings": len(b.recordings),
             "created_at": b.created_at} for b in repo.list_batches()]


@router.post("/batches/{batch_id}/recordings/{recording_id}/upload")
async def upload_recording(batch_id: str, recording_id: str, file: UploadFile = File(...)):
    """Streamed to a temp file in 1 MiB chunks, then atomically renamed; ffprobe runs on the stored file."""
    b = _batch(batch_id)
    r = _rec(b, recording_id)
    if r.state in ("queued", "processing"):
        raise HTTPException(409, f"recording {recording_id} is {r.state}; wait until it finishes")
    dst = repo.media_path(b.id, r.id, r.filename)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    size = 0
    try:
        with tmp.open("wb") as fh:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"recording larger than {MAX_UPLOAD_BYTES} bytes")
                fh.write(chunk)
        if size == 0:
            raise HTTPException(422, "empty file")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    probe = service.ffprobe(dst)
    b = repo.update_recording(b.id, r.id, state="uploaded", stage=None, error=None, size_bytes=size, probe=probe,
                              gx10_path=None, sampled_frames=None, detections=None, tracklets=None,
                              frames_with_person=None, started_at=None, finished_at=None, timings_s={})
    return {"batch_id": b.id, "recording_id": r.id, "state": "uploaded", "size_bytes": size, "probe": probe}


@router.post("/batches/{batch_id}/process")
def process(batch_id: str):
    b = _batch(batch_id)
    if b.status in ("queued", "processing"):
        raise HTTPException(409, f"batch {batch_id} is already {b.status}")
    ready = [r for r in b.recordings if r.state in ("uploaded", "failed")]
    if not ready:
        raise HTTPException(409, "no uploaded recordings to process (upload first; completed ones are not redone)")
    b = service.enqueue(b)
    return {"batch_id": b.id, "status": "queued", "total_recordings": len(b.recordings),
            "queued_recordings": sum(r.state == "queued" for r in b.recordings),
            "not_uploaded": [r.id for r in b.recordings if r.state == "created"]}


@router.post("/batches/{batch_id}/recordings/{recording_id}/retry")
def retry(batch_id: str, recording_id: str):
    b = _batch(batch_id)
    r = _rec(b, recording_id)
    if r.state != "failed":
        raise HTTPException(409, f"only failed recordings can be retried (state: {r.state})")
    repo.update_recording(b.id, r.id, state="queued", stage=None, error=None)
    with repo._lock:
        cur = repo.load_batch(b.id)
        if cur.status not in ("queued", "processing"):
            cur.status, cur.finished_at = "queued", None
        repo.save_batch(cur)
    service.ensure_worker()
    service._q.put(b.id)
    return {"batch_id": b.id, "recording_id": r.id, "state": "queued"}


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str):
    service.ensure_worker()  # resumes work interrupted by a restart
    return service.progress(_batch(batch_id))


@router.get("/batches/{batch_id}/tracklets")
def list_tracklets(batch_id: str):
    _batch(batch_id)
    return [t.model_dump() for t in repo.load_tracklets(batch_id)]


@router.post("/batches/{batch_id}/search")
def search(batch_id: str, body: SearchRequest):
    b = _batch(batch_id)
    try:
        return service.search(b, body)
    except service.gpu_client.GPUWorkerError as e:
        raise HTTPException(e.status if e.status in (404, 422, 503) else 502, f"CLIP text encoder: {e.detail}")


@router.get("/searches/{search_id}")
def get_search(search_id: str):
    s = repo.load_search(search_id)
    if s is None:
        raise HTTPException(404, f"search {search_id} not found")
    return s


@router.get("/tracklets/{tracklet_id}/thumbnail")
def thumbnail(tracklet_id: str):
    if not repo.valid_id(tracklet_id):
        raise HTTPException(404, "tracklet not found")
    p = repo.thumb_path(tracklet_id)
    if not p.is_file():
        raise HTTPException(404, f"no thumbnail for tracklet {tracklet_id}")
    return FileResponse(p, media_type="image/jpeg")


def _find_recording(recording_id: str):
    for b in repo.list_batches():
        for r in b.recordings:
            if r.id == recording_id:
                return b, r
    return None, None


@router.get("/recordings/{recording_id}/playback")
def playback(recording_id: str, request: Request, start_ms: int = 0):
    """Authenticated source recording. HTTP Range is handled by Starlette's FileResponse (seekable in a <video>
    element). start_ms is echoed back in X-Start-Ms (and as #t= in the response JSON contract) for the player."""
    b, r = _find_recording(recording_id) if repo.valid_id(recording_id) else (None, None)
    if r is None:
        raise HTTPException(404, f"recording {recording_id} not found")
    p = repo.media_path(b.id, r.id, r.filename)
    if not p.is_file():
        raise HTTPException(404, f"recording {recording_id} has not been uploaded")
    media = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
             ".webm": "video/webm"}.get(p.suffix, "application/octet-stream")
    return FileResponse(p, media_type=media, filename=Path(r.filename).name,
                        content_disposition_type="inline", headers={"X-Start-Ms": str(max(0, start_ms))})
