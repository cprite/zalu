# ZALU

**Zero-to-model Automated Labelling Unit.** Describe in one sentence what your camera should detect. ZALU labels your
footage automatically, trains a small vision model on a GPU in about a minute, and shows you honestly how well it works
on footage it has never seen.

Imperial AI Hackathon 2026 · Track 02: Voice, Vision & Generative Media · Team MIRAZH (Nikolay, Damir, Kirill).

## Try it live

**👉 https://nikopi5.tail4aaac7.ts.net:8443/**

This is the real product, not a mock-up. Everything you do there runs live on our own remote GPU server (an NVIDIA
GX10 with a GB10 chip): auto-labelling, training, evaluation and inference. Nothing is pre-recorded.

1. Download the sample videos from [`examples/`](examples/README.md) (or use your own footage).
2. Open the link, type a prompt such as `Alert me when a person on site is not wearing a hard hat` and attach
   `examples/helmet/helmet-train.mp4`.
3. Follow the six steps. Training takes about a minute; on Verify upload `examples/helmet/helmet-test.mp4`.

The full walkthrough is in [`examples/README.md`](examples/README.md). It is one GPU shared by everyone, so if
someone else is training at the same moment your job waits in the queue for a minute.

---

## How it works

| Step | What happens | Under the hood |
|---|---|---|
| 1. Describe | "Alert me when a person on site is not wearing a hard hat" | Claude turns the prompt into a task spec: classes, negative classes, the alert event |
| 2. Upload | Attach a few minutes of camera footage | Frames are sampled and split into train / validation |
| 3. Prepare | Every frame is labelled automatically; you fix what is wrong | Open-vocabulary detector (YOLO-World) labels from the text prompt |
| 4. Train | A small model trains on the GPU | Ultralytics YOLO11n, ~30 s for 30–60 epochs on an NVIDIA GB10 |
| 5. Verify | Upload a separate test video | Fine-tuned model vs zero-shot baseline (YOLOE) on unseen footage: precision, recall, F1, FPS, failure gallery, "ready / not ready" verdict |
| 6. Deploy | Download ONNX or run live inference | ONNX export, checked against the PyTorch model |

Plus two features built on the same backend:

- **AI dashboards:** describe the dashboard you want ("count people without hard hats per minute"); Claude plans it
  from the video results.
- **AI Video Search:** upload up to 10 camera recordings, type "worker in a black jacket and a white helmet", get every
  appearance with camera, timestamps and visible duration. Zero-shot, no training.

## Results we measured (real runs, 14–15 Nov 2026)

- Prompt → trained model with ONNX: **63 s** through the API on the final build (15 Nov, hard-hat example, 30 epochs), **~3.5 min** clicking through the UI.
- Hard-hat scenario: **F1 0.49–0.50** (trained YOLO11n) vs **0.13** (zero-shot YOLOE), **213 vs 41 FPS** on the GX10.
- Video Search: 4 recordings indexed in **71 s**, text search over 169 tracks in **78 ms**.
- Honest limits: test labels are auto-generated unless you review them; recall on "no hard hat" is low and the report
  says "not ready"; video search scores are raw CLIP similarity and the same person is not linked across cameras.

---

## Models used

| Model | Used for | Where it runs |
|---|---|---|
| **Anthropic Claude** (via the Claude Code CLI, structured JSON output) | Task Engine: prompt → task spec; Dashboard Planner: prompt → dashboard spec | nikopi5 (app server) |
| **YOLO-World v2** (`yolov8s-worldv2.pt`, Ultralytics) | Auto-labelling frames from the text prompt | NVIDIA GX10 |
| **YOLOE** (`yoloe-11l-seg.pt`, Ultralytics) | Zero-shot baseline in Verify; person detector in Video Search | NVIDIA GX10 |
| **YOLO11n** (`yolo11n.pt`, Ultralytics) | The small model ZALU fine-tunes and deploys (exported to ONNX) | trained on GX10, runs anywhere ONNX runs |
| **CLIP ViT-B/32** (OpenAI) | Video Search: image and text embeddings | NVIDIA GX10 |

## Datasets and footage

| Data | Used for | Licence |
|---|---|---|
| Your own camera footage | Training and testing (that is the product) | yours |
| [Intel IoT DevKit sample videos](https://github.com/intel-iot-devkit/sample-videos) (worker zone, people detection, classroom, store aisle) | Demo scenarios in `examples/` | CC BY 4.0 |
| [Mixkit](https://mixkit.co) free stock clips (construction, warehouse, factory workers) | Demo scenarios in `examples/` | Mixkit free licence |
| Pretrained weights: COCO (YOLO11n), Objects365/GoldG (YOLO-World), CLIP | Starting points; nothing is trained on these by ZALU | Ultralytics AGPL-3.0 / OpenAI MIT |
| [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) | Backend smoke tests | CC BY 4.0 |

No dataset is shipped with test frames that also appear in training: the backend refuses a split where the same image
(by path or hash) is in two splits.

## APIs

### External services

| Service | Used for |
|---|---|
| Claude (Anthropic) via Claude Code CLI | Task Engine and Dashboard Planner |
| Figma | Design tokens synced into the frontend (`design/`) |

### ZALU REST API

FastAPI, bearer-token auth. Full details: [`backend/README.md`](backend/README.md).

| Area | Endpoints |
|---|---|
| Tasks | `POST /api/tasks` (prompt → task spec), `GET /api/tasks/{id}` |
| Datasets | `POST /api/datasets` (upload video/images → auto-labelled dataset), `GET /api/datasets[/{id}]`, `GET …/annotations`, `POST …/labels` (corrections), `POST /api/datasets/register` |
| Training | `POST /api/training`, `GET /api/jobs[/{id}]` |
| Models | `GET /api/models[/{id}]`, `POST /api/models/zero-shot`, `GET /api/models/{id}/onnx` |
| Evaluation | `POST /api/evaluations` (trained vs baseline on a test set), `GET /api/reports/{id}` |
| Inference | `POST /api/inference` (one frame → detections + alert events) |
| Dashboard data | `POST /api/runs` (model over a video), `GET /api/runs/{id}/summary`, `/timeseries`, `/events`, `GET /api/dashboard/sources` |
| AI dashboards | `POST /api/dashboards`, `POST /api/dashboards/{id}/refine`, `GET /api/dashboards/{id}` |
| Video Search | `POST /api/video-search/batches`, `…/recordings/{id}/upload`, `…/process`, `POST …/search`, `GET /api/video-search/tracklets/{id}/thumbnail`, `GET /api/video-search/recordings/{id}/playback` |
| Files / health | `GET /api/files`, `GET /api/health` |

## Architecture

```
Browser (Next.js UI) ──► App API (FastAPI, Raspberry Pi 5 "nikopi5")  ──ssh tunnel──►  GPU worker (FastAPI in Docker, NVIDIA GX10 / GB10)
                          tasks, metadata, Claude CLI, auth                              labelling, training, evaluation, inference, CLIP
```

- `frontend/`: Next.js + React + TypeScript UI (six workflow screens, dashboard, settings, accessibility).
- `backend/`: app API (`main.py`), GPU worker (`gpu_worker.py`), Task / Data / Training / Evaluation / Deployment engines, Video Search.
- `shared/`: shared contracts (DashboardSpec v1).
- `design/`: Figma tokens and plugin.
- `tests/`, `tools/`: end-to-end API tests, timed workflow, webcam client.

## Run it yourself

See [`backend/README.md`](backend/README.md) (GPU worker: `backend/run_gx10.sh`; app API: `backend/run_nikopi5.sh`)
and [`frontend/README.md`](frontend/README.md).
