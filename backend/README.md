# Backend (owner: Kolyan)

FastAPI monolith: Orchestration, Training and Deployment engines, with hooks for the Task/Data Engine (Kirill)
and the Evaluation Engine (Damir).

**Data contracts (agreed 2026-11-14):** `TaskSpec` = `backend/task_engine/schema.py`, `Annotation` / `AnnotationSet` /
`BoxLabel` / `LabelCorrection` / `DatasetManifest` = `backend/data_engine/schema.py` (Kirill's, v1.0).
`backend/schemas.py` re-exports them and adds the backend's own `TrainingRequest`, `TrainingJob`, `ModelArtifact`,
`InferenceResult` and a placeholder `EvaluationReport`. Ids: `task_id`, `dataset_id`, `job_...`, `model_...`.

## Plug-in hooks (until they exist the route returns 501 or a marked MOCK)
| Module | Function the API calls | Owner |
|---|---|---|
| `backend/task_engine/__init__.py` | `create_task_spec(prompt: str) -> TaskSpec` | Kirill |
| `backend/data_engine/__init__.py` | `process_upload(task: TaskSpec, files: list[str]) -> DatasetManifest` (YOLO dataset on disk) | Kirill |
| `backend/data_engine/__init__.py` | `save_corrections(ds: DatasetManifest, labels: list[LabelCorrection]) -> dict` | Kirill |
| `backend/evaluation_engine/__init__.py` | `evaluate(req, models: list[ModelArtifact], test: DatasetManifest) -> EvaluationReport` | Damir |

`GET /api/health` shows which hooks are loaded. Mocks while a hook is missing: `POST /api/tasks` returns
Kirill's `examples/vest.json` spec (compiler `MOCK:...`); `POST /api/datasets` stores the uploads and returns the
coco8 sample manifest (warning `MOCK: ...`); `POST /api/evaluations` returns `mock: true` with the training run's
VAL metrics (never invented numbers).

Every dataset passes a split check before it is stored or trained: no image (by path or by content hash) may be in
two of train/val/test. Failing datasets get HTTP 422 / CLI exit code 2.

## Training CLI (no API needed)
```bash
docker exec -w /data zalu-backend /opt/comfyui-env/bin/python -m backend.training_engine.cli --data coco8.yaml --epochs 10
docker exec -w /data zalu-backend /opt/comfyui-env/bin/python -m backend.training_engine.cli --data X/data.yaml --baseline-only
```
Prints `ZALU_RESULT {json}`: split check, baseline (pretrained, no fine-tuning) metrics, fine-tuned metrics,
train_seconds, ONNX check, paths. Benchmarks: `backend/BENCHMARK.md`.

## API keys for Kirill's engines
Put `ANTHROPIC_API_KEY=...` and `FAL_KEY=...` in `~/.zalu_env` on the GX10 (`chmod 600`); `run_gx10.sh` passes the
file with `--env-file` when it exists. Never commit keys.

## API (all except /api/health need `Authorization: Bearer $ZALU_TOKEN`)
| Method | Path | Notes |
|---|---|---|
| GET | /api/health | CUDA, device, hooks |
| POST | /api/tasks | `{"prompt": "..."}` → TaskSpec (MOCK until Task Engine) |
| POST | /api/datasets | multipart `task_id` + `files[]` → DatasetManifest (needs Data Engine) |
| POST | /api/datasets/register | `{"data_yaml": "/data/...", "task_id"?, "variant"?}` or a full DatasetManifest |
| POST | /api/datasets/{id}/labels | label corrections (Data Engine) |
| POST | /api/training | `{"dataset_id", "base_model": "yolo11n.pt", "epochs", "imgsz", "batch"}` → TrainingJob |
| GET | /api/jobs/{id} | status queued/running/succeeded/failed, progress 0..1, message, error, result_id |
| POST | /api/models/zero-shot | `{"classes": [...]}` → pretrained open-vocabulary baseline (YOLOE) |
| GET | /api/models, /api/models/{id} | metrics + baseline_metrics (val split), train_seconds, weights, onnx, export_check |
| GET | /api/models/{id}/onnx | download the ONNX file (header X-Class-Names) |
| POST | /api/evaluations | `{"model_ids": [...], "test_dataset_id"}` (Evaluation Engine) |
| GET | /api/reports/{id} | EvaluationReport |
| POST | /api/inference | multipart `model_id`, `image`, optional `conf`, `target_event` → detections, latency_ms, events |

One GPU job at a time; each training runs in a child process, logs in `/data/runs/<job_id>.log`.

## Run on the GX10
```bash
rsync -a --exclude .git ./ gx10:zalu/ && ssh gx10 'cd zalu && ZALU_TOKEN=... ./backend/run_gx10.sh'
docker exec -e ZALU_TOKEN=... zalu-backend /opt/comfyui-env/bin/python -m tests.e2e_backend
```

## Measured on the GX10 (2026-11-14, coco8 smoke test, not a benchmark of our task)
- Training yolo11n, 10 epochs on 4 images: 6.9 s; ONNX export included.
- Inference via API, warm: ~15-17 ms per 810x1080 image (trained yolo11n and zero-shot YOLOE-L). First call ~1 s (model load).
- Restaurant clip, 1 h of 1080p video: every frame + tracking 15 min, 1 frame/s 2.7 min (see research notes).

## Live camera on the demo laptop
`tools/webcam_client.py --onnx best.onnx --names a,b` (offline) or `--api URL --model MODEL_ID` (through the backend).
