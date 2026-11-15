"""GX10 GPU worker: the ONLY thing that runs on the GX10. No public API, no LLM, no keys.

Bound to 127.0.0.1 on the GX10; the app on nikopi5 reaches it through an ssh tunnel (that tunnel is the auth).
Everything that needs the GPU or the image files runs here: Data Engine ingestion + YOLO-World auto-labelling,
training (queued, one job at a time, child process), validation, inference, file downloads.

Run (GX10): docker run ... zalu-gpu  (see backend/run_gx10.sh)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend import orchestrator, schemas
from backend.infrastructure import storage
from backend.infrastructure.datasets import manifest_from_yaml
from backend.training_engine.splits import check_splits

app = FastAPI(title="ZALU GPU worker (GX10)", version="0.2.0")


@app.on_event("startup")
def _startup() -> None:
    orchestrator.start()


@app.get("/health")
def health():
    import torch

    return {"ok": True, "role": "gpu-worker", "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "data_root": str(storage.DATA_ROOT)}


# ---------- files ----------
@app.post("/upload")
async def upload(subdir: str = Form(...), files: list[UploadFile] = File(...)):
    if ".." in subdir or subdir.startswith("/"):
        raise HTTPException(422, "bad subdir")
    d = storage.path("uploads", subdir, "x").parent
    saved = []
    for f in files:
        p = d / Path(f.filename or f"upload_{int(time.time())}.bin").name
        p.write_bytes(await f.read())
        saved.append(str(p))
    return {"paths": saved}


@app.get("/files")
def get_file(path: str):
    p = Path(path).resolve()
    if storage.DATA_ROOT not in p.parents or not p.is_file():
        raise HTTPException(404, "file not found under the data root")
    return FileResponse(p, filename=p.name)


# ---------- datasets (Kirill's Data Engine runs here, next to the files and the GPU) ----------
@app.post("/datasets/process")
async def datasets_process(body: dict):
    """body: {task: TaskSpec, files: [paths on GX10], kind: "train"|"test", held_out_sessions?, val_sessions?}"""
    from backend.data_engine import process_test_upload, process_upload

    task = schemas.TaskSpec(**body["task"])
    files = body["files"]
    try:
        fps, mx = float(body.get("fps", 1.0)), int(body.get("max_frames", 200))
        if body.get("kind") == "test":
            m = await run_in_threadpool(process_test_upload, task, files, fps, mx)
        else:
            m = await run_in_threadpool(process_upload, task, files, body.get("held_out_sessions") or (),
                                        body.get("val_sessions"), fps, mx)
    except (ValueError, FileExistsError) as e:
        raise HTTPException(422, str(e))
    check = check_splits(m.data_yaml, m.variant)
    return {"manifest": m.model_dump(), "split_check": check}


@app.post("/datasets/register")
def datasets_register(body: dict):
    if "class_names" in body and "splits" in body:
        m = schemas.DatasetManifest(**body)
        check = check_splits(m.data_yaml)
    else:
        if not body.get("data_yaml") or not (Path(body["data_yaml"]).exists() or body["data_yaml"].endswith(".yaml")):
            raise HTTPException(422, f"data_yaml not found on the GX10: {body.get('data_yaml')}")
        m, check = manifest_from_yaml(body["data_yaml"], body.get("task_id"), body.get("variant", "real_only"),
                                      note=body.get("note") or "registered via /api/datasets/register")
    return {"manifest": m.model_dump(), "split_check": check}


@app.post("/datasets/labels")
def datasets_labels(body: dict):
    from backend.data_engine import save_corrections

    m = schemas.DatasetManifest(**body["manifest"])
    labels = [schemas.LabelCorrection(**x) for x in body["labels"]]
    return save_corrections(m, labels)


@app.post("/datasets/annotations")
def datasets_annotations(body: dict):
    from backend.data_engine import get_annotations

    return get_annotations(schemas.DatasetManifest(**body["manifest"])).model_dump()


_fal_lock = __import__("threading").Lock()


@app.post("/datasets/synthesize")
def datasets_synthesize(body: dict):
    """Kirill's fal synthetic generation on TRAIN frames only (val/test frames are never parents).
    body: {manifest, fal_key, max_images?, conditions?}. The fal key is used for this call only and never stored."""
    import json as _json

    from backend.data_engine import service
    from backend.data_engine.schema import AnnotationSet, FrameCollection
    from backend.data_engine.synthetic import generate_synthetic

    m = schemas.DatasetManifest(**body["manifest"])
    work = Path(m.root).parent / "work"
    if not (work / "annotations.json").exists():
        raise HTTPException(422, f"dataset {m.dataset_id} was not produced by the Data Engine")
    task = schemas.TaskSpec.model_validate_json((work / "task_spec.json").read_text())
    frames = FrameCollection.model_validate_json((work / "frames.json").read_text())
    aset = AnnotationSet.model_validate_json((work / "annotations.json").read_text())
    train_ids = {im.frame_id for im in m.images if im.split == "train"}
    train_frames = [f for f in frames.frames if f.frame_id in train_ids]
    if not train_frames:
        raise HTTPException(422, "no train frames in this dataset")
    out = work / f"synthetic_{int(time.time())}"
    with _fal_lock:
        os.environ["FAL_KEY"] = body["fal_key"]
        try:
            syn = generate_synthetic(aset, train_frames, task, out, detector=service._get_detector(),
                                     conditions=body.get("conditions"), max_images=int(body.get("max_images", 4)))
        finally:
            os.environ.pop("FAL_KEY", None)
    return {"out_dir": str(out), "editor": syn.editor, "stats": syn.annotations.stats,
            "kept": [{"frame_id": f.frame_id, "path": f.path, "parent": f.parent_frame_id,
                      "condition": (f.generation or {}).get("condition")} for f in syn.frames],
            "rejected": [r.model_dump() for r in syn.rejected],
            "parents_pool": {"train_frames": len(train_frames), "excluded_non_train": len(frames.frames) - len(train_frames)}}


# ---------- training ----------
@app.post("/train")
def train(body: dict):
    """body: {request: TrainingRequest, dataset: DatasetManifest}. The manifest is stored here too."""
    ds = schemas.DatasetManifest(**body["dataset"])
    storage.save("datasets", ds)
    req = schemas.TrainingRequest(**body["request"])
    return orchestrator.submit_training(req).model_dump()


@app.get("/jobs/{job_id}")
def job(job_id: str):
    j = storage.load("jobs", job_id, schemas.TrainingJob)
    if j is None:
        raise HTTPException(404, f"job {job_id} not found")
    model = storage.load("models", j.result_id, schemas.ModelArtifact) if j.result_id else None
    return {"job": j.model_dump(), "model": model.model_dump() if model else None}


# ---------- inference ----------
@app.post("/infer")
async def infer(model: str = Form(...), image: UploadFile = File(...), conf: float = Form(0.25),
                target_event: Optional[str] = Form(None)):
    import cv2

    from backend.deployment_engine import inference as inf

    art = schemas.ModelArtifact.model_validate_json(model)
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "could not decode the image (send JPEG or PNG)")
    return (await run_in_threadpool(inf.predict, art, frame, conf, target_event)).model_dump()


# ---------- video inference (dashboard data) ----------
@app.post("/video_infer")
def video_infer(body: dict):
    """body: {model: ModelArtifact, path: video on the GX10, fps: frames per second to sample, conf}.
    Runs the model on sampled frames; returns per-frame detections with video timestamps."""
    import cv2
    import statistics

    from backend.deployment_engine import inference as inf

    art = schemas.ModelArtifact(**body["model"])
    p = Path(body["path"]).resolve()
    if storage.DATA_ROOT not in p.parents or not p.is_file():
        raise HTTPException(404, "video not found under the data root")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise HTTPException(422, "cannot open the video")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(vfps / float(body.get("fps", 2.0))))
    conf = float(body.get("conf", 0.25))
    frames, lat, i, t0 = [], [], 0, time.time()
    w = h = None
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                h, w = f.shape[:2]
                r = inf.predict(art, f, conf)
                lat.append(r.latency_ms)
                frames.append({"frame_index": i, "t_s": round(i / vfps, 3),
                               "detections": [d.model_dump() for d in r.detections]})
        i += 1
    return {"frames": frames, "duration_s": round((n or i) / vfps, 2), "width": w, "height": h,
            "video_fps": vfps, "gpu_ms_median": round(statistics.median(lat), 1) if lat else None,
            "wall_s": round(time.time() - t0, 2)}


# ---------- CLIP (video search): image + text embeddings, same ultralytics CLIP fork YOLO-World uses ----------
_clip_lock = __import__("threading").Lock()
_clip_cache: dict = {}
CLIP_MODEL = os.environ.get("ZALU_CLIP_MODEL", "ViT-B/32")


def _clip():
    import clip
    import torch

    with _clip_lock:
        if "m" not in _clip_cache:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # zalu-gpu has the weights in /data/weights/clip (no internet on the GX10); fall back to ~/.cache/clip
            root = os.environ.get("ZALU_CLIP_DIR", "/data/weights/clip")
            m, pre = clip.load(CLIP_MODEL, device=dev, download_root=root if os.path.isdir(root) else None)
            m.eval()
            _clip_cache.update(m=m, pre=pre, dev=dev)
    return _clip_cache["m"], _clip_cache["pre"], _clip_cache["dev"]


@app.post("/clip/image")
async def clip_image(images: list[UploadFile] = File(...)):
    """JPEG/PNG crops -> L2-normalised CLIP image embeddings, in input order."""
    import io

    import torch
    from PIL import Image

    raw = [await f.read() for f in images]

    def run():
        m, pre, dev = _clip()
        out = []
        for i in range(0, len(raw), 64):
            try:
                x = torch.stack([pre(Image.open(io.BytesIO(b)).convert("RGB")) for b in raw[i:i + 64]]).to(dev)
            except Exception as e:
                raise HTTPException(422, f"could not decode image in chunk {i}: {e}")
            with torch.no_grad():
                e = m.encode_image(x).float()
            out += torch.nn.functional.normalize(e, dim=-1).cpu().tolist()
        return out

    return {"model": CLIP_MODEL, "embeddings": await run_in_threadpool(run)}


@app.post("/clip/text")
def clip_text(body: dict):
    """{"texts": [...]} -> L2-normalised CLIP text embeddings."""
    import clip
    import torch

    texts = body.get("texts") or []
    if not texts or not all(isinstance(t, str) for t in texts):
        raise HTTPException(422, "texts: non-empty list of strings required")
    m, _, dev = _clip()
    with torch.no_grad():
        e = m.encode_text(clip.tokenize(texts, truncate=True).to(dev)).float()
    return {"model": CLIP_MODEL, "embeddings": torch.nn.functional.normalize(e, dim=-1).cpu().tolist()}
