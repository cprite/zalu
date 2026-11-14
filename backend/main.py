"""ZALU app backend (FastAPI), hosted on nikopi5. Public through Tailscale Funnel, bearer-token protected.

Split of work:
- nikopi5 (this app): API, auth, metadata store, Task Engine on the owner's Claude subscription (local `claude` CLI).
- GX10 (backend/gpu_worker.py behind an ssh tunnel): everything that needs the GPU or the image files:
  Data Engine ingestion + auto-labelling, training, inference, file downloads.

Routes follow the shared API contract; while a teammate's engine is missing, the route returns a clearly
marked MOCK (never invented numbers).
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend import schemas
from backend.infrastructure import claude_cli, gpu_client, storage

TOKEN = os.environ.get("ZALU_TOKEN", "")

app = FastAPI(title="ZALU backend", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_bearer = HTTPBearer(auto_error=False)


def auth(cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    """Bearer token check; in Swagger (/docs) use the Authorize button."""
    if TOKEN and (cred is None or cred.credentials != TOKEN):
        raise HTTPException(401, "missing or wrong bearer token")


def _hook(module: str, func: str):
    try:
        return getattr(importlib.import_module(module), func)
    except (ImportError, AttributeError):
        return None


def _gpu(fn, *a, **kw):
    """Call the GX10 worker and turn its errors into HTTP errors with the worker's own message."""
    try:
        return fn(*a, **kw)
    except gpu_client.GPUWorkerError as e:
        raise HTTPException(e.status if e.status in (404, 409, 422, 503) else 502, e.detail)


def _get(kind: str, obj_id: str, model):
    obj = storage.load(kind, obj_id, model)
    if obj is None:
        raise HTTPException(404, f"{kind[:-1]} {obj_id} not found")
    return obj


@app.get("/api/health")
def health():
    try:
        gpu = gpu_client.get("/health", timeout=5)
    except gpu_client.GPUWorkerError as e:
        gpu = {"ok": False, "error": e.detail}
    return {
        "ok": True,
        "host": "nikopi5",
        "gpu_worker": gpu,
        "data_root": str(storage.DATA_ROOT),
        "engines": {
            "task_engine": "claude-cli subscription" if claude_cli.available() else "MOCK (no claude CLI login)",
            "data_engine": "GX10 worker",
            "evaluation_engine": "Damir" if _hook("backend.evaluation_engine", "evaluate") else "MOCK",
        },
    }


# ---------- tasks (route: Kolyan, logic: Kirill; runs here on the Claude subscription) ----------
@app.post("/api/tasks", response_model=schemas.TaskSpec, dependencies=[Depends(auth)])
def create_task(body: dict):
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(422, "prompt is required")
    if claude_cli.available() and not body.get("mock"):
        from backend.task_engine.compiler import TaskCompilationError, _placeholder, compile_task

        try:
            spec = compile_task(prompt, llm=claude_cli.claude_cli_backend())
        except TaskCompilationError as e:  # same behaviour as Kirill's create_task_spec
            spec = _placeholder(prompt[:2000], "needs_clarification", [f"Task Engine could not build a valid spec: {e}"],
                                ["Please rephrase the task, naming the objects to detect."], "claude-cli:error")
        except claude_cli.ClaudeCLIError as e:
            raise HTTPException(502, f"Claude CLI failed: {e}")
    else:  # MOCK: Kirill's vest example spec, clearly labelled
        ex = json.loads((Path(__file__).parent / "task_engine" / "examples" / "vest.json").read_text())
        for k in ("prompt", "task_id", "compiler"):
            ex.pop(k, None)
        spec = schemas.TaskSpec(**ex, prompt=prompt, compiler="MOCK:task_engine/examples/vest.json")
    storage.save("tasks", spec)
    return spec


@app.get("/api/tasks/{task_id}", response_model=schemas.TaskSpec, dependencies=[Depends(auth)])
def get_task(task_id: str):
    return _get("tasks", task_id, schemas.TaskSpec)


# ---------- datasets (route: Kolyan, logic: Kirill; executed on the GX10 next to the files) ----------
def _store_manifest(res: dict) -> schemas.DatasetManifest:
    check = res["split_check"]
    if not check["ok"]:
        raise HTTPException(422, f"dataset split check failed: {check['errors']}")
    m = schemas.DatasetManifest(**res["manifest"])
    storage.save("datasets", m)
    return m


@app.post("/api/datasets", response_model=schemas.DatasetManifest, dependencies=[Depends(auth)])
async def create_dataset(task_id: str = Form(...), kind: str = Form("train"), files: list[UploadFile] = File(...)):
    """Upload footage (videos/images) -> frames -> auto-labels -> YOLO dataset. kind=test makes the independent
    test set (never used for training)."""
    task = _get("tasks", task_id, schemas.TaskSpec)
    payload = [("files", (Path(f.filename or "upload.bin").name, await f.read(), f.content_type or "application/octet-stream"))
               for f in files]
    up = _gpu(gpu_client.post, "/upload", data={"subdir": task.task_id}, files=payload, timeout=600)
    res = _gpu(gpu_client.post, "/datasets/process", json={"task": task.model_dump(), "files": up["paths"], "kind": kind},
               timeout=1800)
    return _store_manifest(res)


@app.post("/api/datasets/register", response_model=schemas.DatasetManifest, dependencies=[Depends(auth)])
def register_dataset(body: dict):
    """Register a YOLO dataset already on the GX10: {"data_yaml": "/data/...", "task_id"?, "variant"?}
    or a full DatasetManifest. Sample datasets by name work too ("coco8.yaml")."""
    return _store_manifest(_gpu(gpu_client.post, "/datasets/register", json=body, timeout=300))


@app.get("/api/datasets", response_model=list[schemas.DatasetManifest], dependencies=[Depends(auth)])
def list_datasets():
    return storage.list_all("datasets", schemas.DatasetManifest)


@app.get("/api/datasets/{dataset_id}", response_model=schemas.DatasetManifest, dependencies=[Depends(auth)])
def get_dataset(dataset_id: str):
    return _get("datasets", dataset_id, schemas.DatasetManifest)


@app.get("/api/datasets/{dataset_id}/annotations", dependencies=[Depends(auth)])
def get_annotations(dataset_id: str):
    """AnnotationSet for Damir's review screen (image_path values can be fetched with GET /api/files)."""
    ds = _get("datasets", dataset_id, schemas.DatasetManifest)
    return _gpu(gpu_client.post, "/datasets/annotations", json={"manifest": ds.model_dump()})


@app.post("/api/datasets/{dataset_id}/labels", dependencies=[Depends(auth)])
def save_labels(dataset_id: str, labels: list[schemas.LabelCorrection]):
    ds = _get("datasets", dataset_id, schemas.DatasetManifest)
    res = _gpu(gpu_client.post, "/datasets/labels",
               json={"manifest": ds.model_dump(), "labels": [x.model_dump() for x in labels]}, timeout=600)
    if res.get("manifest"):
        storage.save("datasets", schemas.DatasetManifest(**res["manifest"]))
    return res


@app.post("/api/datasets/{dataset_id}/synthesize", dependencies=[Depends(auth)])
def synthesize(dataset_id: str, body: dict = {}):
    """fal synthetic variants of TRAIN frames (paid: ~$0.04 per image). body: {max_images: 4, conditions?: [...]}.
    The FAL_KEY lives on nikopi5 and is sent to the GPU worker for this call only."""
    key = os.environ.get("FAL_KEY")
    if not key:
        raise HTTPException(503, "FAL_KEY is not configured on nikopi5")
    n = int(body.get("max_images", 4))
    if not 1 <= n <= 20:
        raise HTTPException(422, "max_images must be 1..20 (each image is a paid fal call)")
    ds = _get("datasets", dataset_id, schemas.DatasetManifest)
    return _gpu(gpu_client.post, "/datasets/synthesize", timeout=1800,
                json={"manifest": ds.model_dump(), "fal_key": key, "max_images": n, "conditions": body.get("conditions")})


@app.get("/api/files", dependencies=[Depends(auth)])
def get_file(path: str):
    """Any file under the GX10 data root (frames, previews, ONNX), streamed through nikopi5."""
    r = _gpu(gpu_client.stream_file, path)
    return StreamingResponse(r.iter_content(64 * 1024), media_type=r.headers.get("content-type", "application/octet-stream"))


# ---------- training + jobs (Kolyan; runs on the GX10) ----------
@app.post("/api/training", response_model=schemas.TrainingJob, dependencies=[Depends(auth)])
def start_training(req: schemas.TrainingRequest):
    ds = _get("datasets", req.dataset_id, schemas.DatasetManifest)
    if ds.variant == "test":
        raise HTTPException(422, "this is an independent TEST dataset: it can never be used for training")
    job = schemas.TrainingJob(**_gpu(gpu_client.post, "/train", json={"request": req.model_dump(), "dataset": ds.model_dump()}))
    storage.save("jobs", job)
    return job


@app.get("/api/jobs/{job_id}", response_model=schemas.TrainingJob, dependencies=[Depends(auth)])
def get_job(job_id: str):
    local = _get("jobs", job_id, schemas.TrainingJob)
    if local.status in (schemas.JobStatus.succeeded, schemas.JobStatus.failed):
        return local
    res = _gpu(gpu_client.get, f"/jobs/{job_id}")
    job = schemas.TrainingJob(**res["job"])
    storage.save("jobs", job)
    if res.get("model"):
        storage.save("models", schemas.ModelArtifact(**res["model"]))
    return job


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
    art = _get("models", model_id, schemas.ModelArtifact)
    if not art.onnx:
        raise HTTPException(404, f"model {model_id} has no ONNX export")
    r = _gpu(gpu_client.stream_file, art.onnx)
    return StreamingResponse(r.iter_content(64 * 1024), media_type="application/octet-stream",
                             headers={"Content-Disposition": f'attachment; filename="{model_id}.onnx"',
                                      "X-Class-Names": ",".join(art.classes)})


# ---------- evaluation (route: Kolyan, logic: Damir) ----------
@app.post("/api/evaluations", response_model=schemas.EvaluationReport, dependencies=[Depends(auth)])
def evaluate(req: schemas.EvaluationRequest):
    models = [_get("models", i, schemas.ModelArtifact) for i in req.model_ids]
    test = _get("datasets", req.test_dataset_id, schemas.DatasetManifest)
    fn = _hook("backend.evaluation_engine", "evaluate")
    if fn is None:  # MOCK until Damir's engine lands. No invented numbers: copies val-split metrics.
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


# ---------- inference (Kolyan; runs on the GX10) ----------
@app.post("/api/inference", response_model=schemas.InferenceResult, dependencies=[Depends(auth)])
async def inference(model_id: str = Form(...), image: UploadFile = File(...), conf: float = Form(0.25),
                    target_event: Optional[str] = Form(None)):
    art = _get("models", model_id, schemas.ModelArtifact)
    data = {"model": art.model_dump_json(), "conf": conf}
    if target_event:
        data["target_event"] = target_event
    files = {"image": (image.filename or "frame.jpg", await image.read(), image.content_type or "image/jpeg")}
    return _gpu(gpu_client.post, "/infer", data=data, files=files, timeout=60)
