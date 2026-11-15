# ZALU frontend: integration guide (for Kolyan and Kirill)

The frontend uses the shared DashboardSpec v1 from `shared/dashboard_spec`.

## 1. Run the frontend
Requirements: Node.js 20+ (tested with Node 24, npm 11).

```bash
git fetch origin
git switch main
cd frontend
npm ci
npm run dev            # http://localhost:3000
```

The app uses the live API by default. To run a local offline demo, set `NEXT_PUBLIC_ZALU_MODE=demo`
in `frontend/.env.local` before starting. Remove the setting before building for production.

## 2. Configure the API

The browser calls same-origin `/api`. Set the server-only `ZALU_API_URL` to the upstream origin
before the build, or have an authenticated deployment gateway serve `/api` directly and leave the
variable unset. For a bearer-protected FastAPI deployment, authenticate users at the gateway and
attach the backend's bearer token there. Keep the credential out of `NEXT_PUBLIC_*`, browser storage
and client code. Only expose authenticated routes to authorised users.

## 3. What each screen calls (live mode)
| Step | Calls | Notes |
|---|---|---|
| Describe | `POST /api/tasks {prompt}` | shows `summary`, `classes[].name`, `visual_conditions`, and `issues` / `clarification_questions` when `status != ok`. `compiler: MOCK…` is shown as sample. |
| Upload | `POST /api/datasets` multipart `task_id`, `kind=train`, `files[]` | videos or images; waits for the GX10 (frames + auto-labels). Up to 1800 s server-side. |
| Prepare | `GET /api/datasets/{id}/annotations`, `GET /api/files?path=<image_path>`, `POST /api/datasets/{id}/labels` | shows up to 60 frames; sends `LabelCorrection` only for frames whose boxes changed (`source: manual` for drawn/confirmed boxes). |
| Train | `GET /api/datasets`, `POST /api/training`, `GET /api/jobs/{id}` (poll 1.5 s), `GET /api/models/{id}`, `POST /api/models/zero-shot` | `variant=test` datasets are never offered for training. |
| Verify | `POST /api/datasets` with `kind=test`, `POST /api/evaluations {model_ids, test_dataset_id}` | if the backend returns `mock: true`, the UI shows val metrics with a "Not an independent test" warning. |
| Deploy | `POST /api/dashboards`, `POST /api/dashboards/{id}/refine`, and multipart `POST /api/inference` | Dashboard planning uses the selected task and trained model. Camera needs permission on HTTPS or localhost. Image upload also works for one-frame inference. |

Conversion between Kirill's contracts (`task_id`, `dataset_id`, `ClassDef`, `BoxLabel`) and the UI model is only in
`frontend/src/lib/api/live.ts`. If a contract changes, that is the one file to update.

## 4. Evaluation Engine (Damir) in this branch
`backend/evaluation_engine/` is picked up by the existing `_hook` in `main.py`, so `/api/health` reports
`"evaluation_engine": "Damir"` and `/api/evaluations` returns real independent-test metrics instead of the mock.
It runs on nikopi5 and uses `backend.infrastructure.gpu_client` only:
- ground truth: `POST /datasets/annotations` (reviewed labels of the `test` split),
- frames: `GET /files`, predictions: `POST /infer` with the model JSON (same path as `/api/inference`).
Leakage checks: test sessions vs train/val sessions of each model's dataset, synthetic frames, `variant`.
Optional: `pip install pillow` on nikopi5 adds failure thumbnails to the report (without it the report still works and
gives `image_path` for each failure).

Tests (no GPU, no network): `python -m pytest tests/test_evaluation.py tests/test_evaluation_api.py -q`

## 5. Known gaps (clearly marked in the UI)
- Real + synthetic training needs a `real_plus_synthetic` dataset; the UI does not call the paid `/synthesize` yet.
- The "Recording name" field only matters in Sample mode; live session ids come from the Data Engine.
- The Data Coach is rule-based on the frames shown; Data Engine suggestions are not wired yet.
- Deploy shows server-side latency; the offline ONNX path is `GET /api/models/{id}/onnx` + `tools/webcam_client.py`.
