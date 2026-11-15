# Evaluation Engine (owner: Damir)

Scores detections on **independent real footage** and records failure examples.
`backend/main.py` discovers `evaluate()` through its existing `_hook`, so `POST /api/evaluations` and
`GET /api/reports/{id}` serve the shared `backend.schemas.EvaluationReport` envelope (`/api/health` shows `"evaluation_engine": "Damir"`).
The evaluation route translates input errors to HTTP 422 and uses the existing GX10 error mapping.

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
| `yolo.py` | YOLO labels and Ultralytics or GX10 inference adapters |
| `engine.py` | the hook: load test split, leakage checks, run models, Failure Gallery |

Tests with synthetic inputs and a fake GX10: `python -m pytest -q tests/test_evaluation.py tests/test_evaluation_yolo.py tests/test_evaluation_api.py`.

## Method (fixed before testing)
- A prediction counts if `score >= conf_threshold` (default 0.25, env `ZALU_EVAL_CONF`); predictions are collected at the lower of 0.05 and the scoring threshold.
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


## YOLO and FastAPI integration

The API takes `{"model_ids":["model_…"],"test_dataset_id":"ds_…"}` at `POST /api/evaluations` and
persists the report for `GET /api/reports/{id}`. The hook loads the exact manifest `test` roster. A
manifest with only validation images gets HTTP 422 before inference. Use per-frame manifest entries
with image and label paths, recording session IDs and review status. Counts-only manifests cannot
supply an evaluation roster.

- Data Engine datasets use the matching AnnotationSet. Every included frame needs an annotation,
  including frames with empty boxes. Rejected annotations are excluded.
- For registered datasets with a full manifest and no AnnotationSet, the hook reads the label files.
  `parse_yolo_labels(image, text, (width, height), class_names)` takes five-column YOLO detection labels
  (`class_id cx cy width height`). Coordinates are normalized against the original image. IDs follow
  the manifest's class order. An empty file identifies a background image. Supply an explicit empty
  file for confirmed background if the exporter omitted it.
- `from_inference_result(image, packet, size=(width, height), model_id=...)` accepts GX10 JSON or
  `schemas.InferenceResult`. The hook checks model ID and dimensions against the decoded frame,
  then passes pixel `xyxy` boxes to the existing evaluator. GX10 rounds scores and coordinates,
  which can affect matches near a boundary.
- `from_ultralytics_result(image, result, size=(width, height))` consumes raw detector Results
  (`orig_shape`, `boxes.xyxy`, `boxes.cls`, `boxes.conf`, `names`) without importing Ultralytics.
  Both adapters accept an optional `class_map` from model labels to manifest labels. Direct Python
  callers of `evaluate` can supply `class_maps={model_id: class_map}`. The HTTP route expects models
  to emit the canonical task labels. The scorer evaluates the full manifest taxonomy and ignores
  predictions outside it. [Ultralytics Results reference](https://docs.ultralytics.com/reference/engine/results/)

To tune confidence, use `dataset_io.load_split(manifest, "val")`, collect validation predictions at
a floor at or below every candidate, then call
`core.select_threshold(gt, preds, images, classes, split="val")`. Freeze the model artifacts,
taxonomy and thresholds before opening held-out labels or running final inference. The API uses
the existing `ZALU_EVAL_CONF` and `ZALU_EVAL_IOU` settings for all requested models.

The hook flags overlapping recording sessions, synthetic frames, missing training provenance and
unreviewed ground truth as `invalid_test_set`. Those reports retain diagnostic metrics and suppress
model comparisons. Confirm split independence and artifact selection upstream. Session IDs alone
cannot prove that footage was never used during tuning.

The existing `per_model` fields remain compatible with `frontend/src/lib/types.ts`. Failure entries
include original `width` and `height` in pixels alongside the existing coordinates, image path and
thumbnail. The FastAPI tests round-trip the real response schema, exercise registered YOLO
labels and verify that missing labels and invalid worker dimensions surface as errors.
