# Backend (owner: Kolyan)

## Where things run (2026-11-14, owner's decision)
- **nikopi5**: the app (`backend/main.py`, systemd user unit `zalu-api`, 127.0.0.1:9100, public via Tailscale Funnel
  `https://nikopi5.tail4aaac7.ts.net:10000`). API, auth, metadata (`~/zalu-app/data`), Task Engine on the owner's
  Claude subscription through the local `claude` CLI (no API key). Keys (`ZALU_TOKEN`, `FAL_KEY`) only here.
  Install/restart: `backend/run_nikopi5.sh`.
- **GX10 = external GPU only**: `backend/gpu_worker.py` in Docker `zalu-gpu` (127.0.0.1:9200), reached through the
  `zalu-tunnel` ssh tunnel. Runs everything that needs the GPU or the image files: Data Engine ingestion +
  auto-labelling, training, validation, inference, file downloads. No public port, no LLM, no keys.
  Install/restart: `backend/run_gx10.sh` on the GX10.
- End-to-end test (public URL): `ZALU_API=... ZALU_TOKEN=... python3 tests/e2e_backend.py --footage video.mp4`.

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
docker exec -w /data zalu-gpu /opt/comfyui-env/bin/python -m backend.training_engine.cli --data coco8.yaml --epochs 10
docker exec -w /data zalu-gpu /opt/comfyui-env/bin/python -m backend.training_engine.cli --data X/data.yaml --baseline-only
```
Prints `ZALU_RESULT {json}`: split check, baseline (pretrained, no fine-tuning) metrics, fine-tuned metrics,
train_seconds, ONNX check, paths. Benchmarks: `backend/BENCHMARK.md`.

## Keys for Kirill's engines (`~/.zalu_env` on the GX10, chmod 600, passed with --env-file; never commit)
- `FAL_KEY=...` for synthetic images (set).
- **No Anthropic API key**: the Task Engine runs on the owner's Claude subscription through the Claude Code CLI
  (installed in the image). Auth: `CLAUDE_CODE_OAUTH_TOKEN=...` from `claude setup-token`.
  `backend/infrastructure/claude_cli.py` is a drop-in `LLMBackend` for `compile_task(prompt, llm=...)`;
  `/api/tasks` uses it automatically when the token is present (`GET /api/health` → `claude_cli_subscription: true`),
  otherwise it returns the mock vest spec. Tested on nikopi5: "Detect workers without safety vests" → valid spec in 17 s.

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

## Deploy
```bash
rsync -a --exclude .git ./ gx10:zalu/ && ssh gx10 'cd zalu && ./backend/run_gx10.sh'   # GPU worker
./backend/run_nikopi5.sh                                                               # app on nikopi5
```

## Measured on the GX10 (2026-11-14, coco8 smoke test, not a benchmark of our task)
- Training yolo11n, 10 epochs on 4 images: 6.9 s; ONNX export included.
- Inference via API, warm: ~15-17 ms per 810x1080 image (trained yolo11n and zero-shot YOLOE-L). First call ~1 s (model load).
- Restaurant clip, 1 h of 1080p video: every frame + tracking 15 min, 1 frame/s 2.7 min (see research notes).

## Live camera on the demo laptop
`tools/webcam_client.py --onnx best.onnx --names a,b` (offline) or `--api URL --model MODEL_ID` (through the backend).
