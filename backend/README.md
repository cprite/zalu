# Backend (owner: Kolyan)

FastAPI monolith: Orchestration, Training and Deployment engines, with hooks for the Task/Data Engine (Kirill)
and the Evaluation Engine (Damir). Data contracts: `backend/schemas.py` (PROPOSAL, agree before integrating).

## Plug-in hooks (until they exist the route returns 501 or a marked MOCK)
| Module | Function the API calls | Owner |
|---|---|---|
| `backend/task_engine/__init__.py` | `create_task_spec(prompt: str) -> TaskSpec` | Kirill |
| `backend/data_engine/__init__.py` | `process_upload(task: TaskSpec, files: list[str]) -> DatasetManifest` (YOLO dataset on disk) | Kirill |
| `backend/data_engine/__init__.py` | `save_corrections(ds: DatasetManifest, labels: list[Annotation]) -> dict` | Kirill |
| `backend/evaluation_engine/__init__.py` | `evaluate(req, models: list[ModelArtifact], test: DatasetManifest) -> EvaluationReport` | Damir |

`GET /api/health` shows which hooks are loaded.

## API (all except /api/health need `Authorization: Bearer $ZALU_TOKEN`)
| Method | Path | Notes |
|---|---|---|
| GET | /api/health | CUDA, device, hooks |
| POST | /api/tasks | `{"prompt": "..."}` → TaskSpec (MOCK until Task Engine) |
| POST | /api/datasets | multipart `task_id` + `files[]` → DatasetManifest (needs Data Engine) |
| POST | /api/datasets/register | register an existing YOLO dataset folder (dev) |
| POST | /api/datasets/{id}/labels | label corrections (Data Engine) |
| POST | /api/training | `{"dataset_id", "base_model": "yolo11n.pt", "epochs", "imgsz", "batch"}` → TrainingJob |
| GET | /api/jobs/{id} | status queued/running/succeeded/failed, progress 0..1, message, error, result_id |
| POST | /api/models/zero-shot | `{"classes": [...]}` → pretrained open-vocabulary baseline (YOLOE) |
| GET | /api/models, /api/models/{id} | metrics (val split), train_seconds, weights, onnx |
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
