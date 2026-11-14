"""ZALU / VisionForge backend (FastAPI). Routes follow the shared API contract in the master brief.

Kirill's Task/Data logic and Damir's Evaluation logic plug in through the hooks below; until their modules
exist, those routes return 501 (or a clearly marked mock) so the frontend can be wired up in parallel.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware

from backend import orchestrator, schemas
from backend.infrastructure import storage
from backend.training_engine.splits import check_splits
from backend.infrastructure.datasets import manifest_from_yaml

TOKEN = os.environ.get("ZALU_TOKEN", "")

app = FastAPI(title="ZALU backend", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


_bearer = HTTPBearer(auto_error=False)


def auth(cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    """Bearer token check; in Swagger (/docs) use the Authorize button."""
    if TOKEN and (cred is None or cred.credentials != TOKEN):
        raise HTTPException(401, "missing or wrong bearer token")


def _hook(module: str, func: str):
    """Return a teammate's function if their module exists, else None."""
    try:
        return getattr(importlib.import_module(module), func)
    except (ImportError, AttributeError):
        return None


@app.on_event("startup")
def _startup() -> None:
    orchestrator.start()


@app.get("/api/health")
def health():
    import torch

    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_root": str(storage.DATA_ROOT),
        "hooks": {
            "task_engine": _hook("backend.task_engine", "create_task_spec") is not None,
            "claude_cli_subscription": __import__("backend.infrastructure.claude_cli", fromlist=["x"]).available(),
            "data_engine": _hook("backend.data_engine", "process_upload") is not None,
            "evaluation_engine": _hook("backend.evaluation_engine", "evaluate") is not None,
        },
    }


# ---------- tasks (route: Kolyan, logic: Kirill) ----------
@app.post("/api/tasks", response_model=schemas.TaskSpec, dependencies=[Depends(auth)])
def create_task(body: dict):
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(422, "prompt is required")
    fn = _hook("backend.task_engine", "create_task_spec")
    from backend.infrastructure import claude_cli

    if fn is None and claude_cli.available() and not body.get("mock"):
        # Kirill's compiler on the owner's Claude subscription (claude CLI), no metered API key
        from backend.task_engine.compiler import TaskCompilationError, compile_task

        try:
            spec = compile_task(prompt, llm=claude_cli.claude_cli_backend())
        except (TaskCompilationError, claude_cli.ClaudeCLIError) as e:
            raise HTTPException(502, f"task compilation failed: {e}")
        storage.save("tasks", spec)
        return spec
    if fn is None:  # MOCK: Kirill's vest example spec, labelled as a mock (no claude CLI auth, or body.mock)
        import json

        ex = json.loads((Path(__file__).parent / "task_engine" / "examples" / "vest.json").read_text())
        for k in ("prompt", "task_id", "compiler"):
            ex.pop(k, None)
        spec = schemas.TaskSpec(**ex, prompt=prompt, compiler="MOCK:task_engine/examples/vest.json")
    else:
        spec = fn(prompt)
    storage.save("tasks", spec)
    return spec


@app.get("/api/tasks/{task_id}", response_model=schemas.TaskSpec, dependencies=[Depends(auth)])
def get_task(task_id: str):
    return _get("tasks", task_id, schemas.TaskSpec)


# ---------- datasets (route: Kolyan, logic: Kirill) ----------
@app.post("/api/datasets", response_model=schemas.DatasetManifest, dependencies=[Depends(auth)])
async def create_dataset(task_id: str = Form(...), files: list[UploadFile] = File(...)):
    fn = _hook("backend.data_engine", "process_upload")
    task = _get("tasks", task_id, schemas.TaskSpec)
    upload_dir = storage.path("uploads", task_id, "x").parent
    saved = []
    for f in files:
        p = upload_dir / Path(f.filename or "upload.bin").name
        p.write_bytes(await f.read())
        saved.append(str(p))
    if fn is None:  # MOCK until Kirill's connector lands: uploads are stored, the sample dataset is returned
        manifest, check = manifest_from_yaml("coco8.yaml", task_id=task.task_id,
                                             note=f"MOCK: {len(saved)} upload(s) stored in {upload_dir}, not processed")
    else:
        manifest = fn(task, saved)
        check = check_splits(manifest.data_yaml)
    if not check["ok"]:
        raise HTTPException(422, f"dataset split check failed: {check['errors']}")
    storage.save("datasets", manifest)
    return manifest


@app.post("/api/datasets/register", response_model=schemas.DatasetManifest, dependencies=[Depends(auth)])
def register_dataset(body: dict):
    """Register a YOLO dataset already on the training host.
    Body: either a full DatasetManifest (from the Data Engine) or {"data_yaml": ..., "task_id"?, "variant"?}."""
    if "class_names" in body and "splits" in body:
        m = schemas.DatasetManifest(**body)
        check = check_splits(m.data_yaml)
    else:
        if not body.get("data_yaml") or not Path(body["data_yaml"]).exists():
            raise HTTPException(422, f"data_yaml not found on the training host: {body.get('data_yaml')}")
        m, check = manifest_from_yaml(body["data_yaml"], body.get("task_id"), body.get("variant", "real_only"),
                                      note="registered via /api/datasets/register")
    if not check["ok"]:
        raise HTTPException(422, f"dataset split check failed: {check['errors']}")
    storage.save("datasets", m)
    return m


@app.post("/api/datasets/{dataset_id}/labels", dependencies=[Depends(auth)])
def save_labels(dataset_id: str, labels: list[schemas.LabelCorrection]):
    fn = _hook("backend.data_engine", "save_corrections")
    if fn is None:
        raise HTTPException(501, "label corrections live in the Data Engine (Kirill)")
    return fn(_get("datasets", dataset_id, schemas.DatasetManifest), labels)


@app.get("/api/datasets", response_model=list[schemas.DatasetManifest], dependencies=[Depends(auth)])
def list_datasets():
    return storage.list_all("datasets", schemas.DatasetManifest)


# ---------- training + jobs (Kolyan) ----------
@app.post("/api/training", response_model=schemas.TrainingJob, dependencies=[Depends(auth)])
def start_training(req: schemas.TrainingRequest):
    try:
        return orchestrator.submit_training(req)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/jobs/{job_id}", response_model=schemas.TrainingJob, dependencies=[Depends(auth)])
def get_job(job_id: str):
    return _get("jobs", job_id, schemas.TrainingJob)


@app.get("/api/jobs", response_model=list[schemas.TrainingJob], dependencies=[Depends(auth)])
def list_jobs():
    return storage.list_all("jobs", schemas.TrainingJob)


# ---------- models ----------
@app.post("/api/models/zero-shot", response_model=schemas.ModelArtifact, dependencies=[Depends(auth)])
def create_zero_shot(body: dict):
    """Pretrained baseline for comparisons: open-vocabulary model prompted with the class names."""
    from backend.training_engine.trainer import zero_shot_artifact

    classes = body.get("classes") or []
    if not classes:
        raise HTTPException(422, "classes are required")
    art = zero_shot_artifact(classes)
    storage.save("models", art)
    return art


@app.get("/api/models", response_model=list[schemas.ModelArtifact], dependencies=[Depends(auth)])
def list_models():
    return storage.list_all("models", schemas.ModelArtifact)


@app.get("/api/models/{model_id}", response_model=schemas.ModelArtifact, dependencies=[Depends(auth)])
def get_model(model_id: str):
    return _get("models", model_id, schemas.ModelArtifact)


@app.get("/api/models/{model_id}/onnx", dependencies=[Depends(auth)])
def download_onnx(model_id: str):
    """Exported ONNX weights for offline inference on the demo laptop (tools/webcam_client.py --onnx)."""
    from fastapi.responses import FileResponse

    art = _get("models", model_id, schemas.ModelArtifact)
    if not art.onnx or not Path(art.onnx).exists():
        raise HTTPException(404, f"model {model_id} has no ONNX export")
    names = ",".join(art.classes)
    return FileResponse(art.onnx, media_type="application/octet-stream", filename=f"{model_id}.onnx",
                        headers={"X-Class-Names": names})


# ---------- evaluation (route: Kolyan, logic: Damir) ----------
@app.post("/api/evaluations", response_model=schemas.EvaluationReport, dependencies=[Depends(auth)])
def evaluate(req: schemas.EvaluationRequest):
    fn = _hook("backend.evaluation_engine", "evaluate")
    models = [_get("models", i, schemas.ModelArtifact) for i in req.model_ids]
    test = _get("datasets", req.test_dataset_id, schemas.DatasetManifest)
    if fn is None:  # MOCK until Damir's Evaluation Engine lands. No invented numbers: copies val-split metrics.
        rep = schemas.EvaluationReport(
            request=req, mock=True,
            per_model={m.id: {"kind": m.kind, "variant": m.variant, "val_metrics": m.metrics,
                              "baseline_val_metrics": m.baseline_metrics, "train_seconds": m.train_seconds}
                       for m in models},
            notes=["MOCK report: Evaluation Engine (Damir) not merged.",
                   "Numbers are the training run's VAL-split metrics, not an independent test-set evaluation."])
    else:
        rep = fn(req, models, test)
    storage.save("reports", rep)
    return rep


@app.get("/api/reports/{report_id}", response_model=schemas.EvaluationReport, dependencies=[Depends(auth)])
def get_report(report_id: str):
    return _get("reports", report_id, schemas.EvaluationReport)


# ---------- inference (Kolyan) ----------
@app.post("/api/inference", response_model=schemas.InferenceResult, dependencies=[Depends(auth)])
async def inference(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    target_event: Optional[str] = Form(None),
):
    import cv2

    from backend.deployment_engine import inference as inf

    art = _get("models", model_id, schemas.ModelArtifact)
    buf = np.frombuffer(await image.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "could not decode the image (send JPEG or PNG)")
    return inf.predict(art, frame, conf=conf, target_event=target_event)


def _get(kind: str, obj_id: str, model):
    obj = storage.load(kind, obj_id, model)
    if obj is None:
        raise HTTPException(404, f"{kind[:-1]} {obj_id} not found")
    return obj
