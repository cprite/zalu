# Evaluation Engine (owner: Damir)

Measures a model honestly on **independent real footage** and explains its failures.
`backend/main.py` discovers `evaluate()` through its existing `_hook`, so `POST /api/evaluations` and
`GET /api/reports/{id}` return real test metrics as soon as this folder is present (`/api/health` → `"evaluation_engine": "Damir"`).
No change to `main.py` or the shared schemas.

## Where it runs
On nikopi5 (API host). Everything that needs files or the GPU goes through `backend.infrastructure.gpu_client`:
ground truth `POST /datasets/annotations`, frames `GET /files`, predictions `POST /infer` (same path as `/api/inference`).
On the GX10 itself set `ZALU_EVAL_LOCAL=1` to call the Deployment Engine in-process; files are read locally when present.

## Files
| File | What |
|---|---|
| `matching.py` | IoU, greedy per-class matching, failure reasons (pure Python) |
| `metrics.py` | precision / recall / F1, 95% Wilson intervals, sample-size warnings |
| `core.py` | `evaluate_model`, `compare` (baseline vs fine-tuned, synthetic A/B), `readiness` (acceptance criteria), `select_threshold` (validation only) |
| `dataset_io.py` | test split from Kirill's `DatasetManifest` + `AnnotationSet` (or YOLO labels for registered datasets) |
| `leakage.py` | test-set independence checks (sessions vs train/val, synthetic frames, variant) |
| `engine.py` | the hook: load test split, leakage checks, run models, Failure Gallery |

Tests (no GPU, no network): `python -m pytest tests/test_evaluation.py tests/test_evaluation_api.py -q`.

## Method (fixed before testing)
- A prediction counts if `score >= conf_threshold` (default 0.25, env `ZALU_EVAL_CONF`); predictions are collected at 0.05 to explain misses.
- Matching per class, greedy by descending score, one prediction per object, `IoU >= iou_threshold` (default 0.5, env `ZALU_EVAL_IOU`).
- Box-level micro P/R/F1 plus an image-level view ("was the frame flagged?"). 95% Wilson intervals.
- Speed: median model latency on the GPU worker (first call excluded), not including network.
- Verdict `ready` / `not_ready` / `insufficient_data` / `invalid_test_set` against explicit criteria (recall ≥ 0.80, precision ≥ 0.70,
  FPS ≥ 10, ≥ 20 images, ≥ 30 objects per class). Every summary says a small test set never proves production safety.

## Report shape (`EvaluationReport.per_model[model_id]`)
`overall{tp,fp,fn,precision,recall,f1,precision_ci95,recall_ci95}`, `per_class{...,image_level}`, `test{n_images,n_sessions,gt_per_class,dataset_id,split}`,
`failure_counts`, `failures[]` (max 30, misses first; `image_path` fetchable via `/api/files`, `thumbnail` data URI when Pillow is installed),
`performance`, `warnings[]`, `readiness{verdict,summary,criteria,data_criteria}`, `delta_vs_baseline`, `model{...}`,
flat `precision/recall/f1/fp/fn/fps`, `sample_data`. Frontend types: `frontend/src/lib/types.ts`.
