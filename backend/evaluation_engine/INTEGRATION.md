# ZALU evaluation handoff

## API payload for Kolyan and frontend

The existing backend envelope in `backend/schemas.py` accepts a dictionary keyed by model ID in `per_model`. The proposed integration puts one complete v1.0 EvaluationReport at `per_model[model_id]`. Keep the outer `id`, `request`, `mock`, `notes` and `created_at` fields. The nested report uses its own `schema_version` and `task_type`. For the current mock route, the frontend can check for `schema_version` before reading nested report fields. The [complete synthetic API response](examples/yolo/api_report.synthetic.json) sets `mock: true`.

[EvaluationReport JSON Schema](evaluation_report.schema.json) is Draft 2020-12 and applies to each nested entry. Read these required fields:

| Field | Meaning and UI use |
| --- | --- |
| `schema_version`, `task_type` | `"1.0"` and `"object_detection"` |
| `dataset` | ID, evaluated split, provenance, class roster, counts and fingerprint of the annotated image roster |
| `model`, `config` | Artifact ID, model variant and the actual confidence/IoU thresholds used |
| `summary`, `per_class` | Micro metrics and class metrics with TP/FP/FN and ground-truth support. Precision, recall and F1 are fractions in [0, 1]. |
| `confusion`, `error_counts` | Diagnostic rows and counts. A null class represents a missed prediction or an unmatched prediction against background. |
| `failures` | Frame ID, original width and height, ground truth and scored predictions in pixel `xyxy`. Error indices are zero-based into the arrays on that frame. An absent box index is null. |
| `readiness` | Policy, observed values and check outcomes. With a policy, only independently labelled real test data can be `passed` or `failed`. |

For overlays, resolve `image_id` through the authorized dataset manifest file lookup. The example contains synthetic IDs and coordinates.

## Input and Python functions

Import the adapter functions from `backend.evaluation_engine.yolo`.

- `load_yolo_ground_truth(images, dataset_id=..., split=..., source=..., class_names=...)` reads a roster of `YoloLabelImage(image_id, label_path, width, height)`. Each label file contains one `class_id cx cy width height` row per object. Classes are zero-based indices into the dataset's ordered class list. Coordinates are normalized to the original image. The adapter converts center-size coordinates to pixel corners, with a 1e-6 normalized tolerance for edge rounding in the Data Engine's six-decimal exports. An empty file marks reviewed background. Create an explicit empty file for a confirmed background image if its upstream YOLO export omits the label.
- `from_ultralytics_result(image_id, result, class_map=...)` reads detection `Results.orig_shape` in height-width order, `boxes.xyxy`, `boxes.cls`, `boxes.conf` and `result.names`. Supply the manifest frame ID explicitly and use dimensions from the original decoded frame. `class_map` maps model labels to canonical task labels. Dataset and model class orders can differ. Unmapped labels stay in the report and count as FPs if outside the task taxonomy. [Ultralytics Results reference](https://docs.ultralytics.com/reference/engine/results/)
- `from_inference_result(image_id, result, class_map=...)` reads the current GX10 `/infer` JSON or a validated `InferenceResult.model_dump()`. Detection fields are `cls`, `box` (pixel `xyxy`) and `score`.
- `evaluate_yolo(ground_truth, prediction_rows, config, model_id=..., variant=..., readiness=...)` requires one inference row per roster frame, matching image dimensions and the same pixel coordinate system. A successful inference with no detections supplies an empty row. It returns a complete EvaluationReport.
- `build_test_per_model(reports, dataset_id=..., model_ids=...)` packages independently labelled real test reports for the matching request. It checks model IDs, dataset identity and fingerprint, complete prediction rows and shared IoU. Use `compare_models(a, b)` for deltas between two reports.

## Wire to the current nikopi5/GX10 backend

On validation, pass `split="validation"` for the manifest's `val` split and run `select_on_validation(reports)` from `backend.evaluation_engine`. Finalize model IDs, model versions, class maps, confidence thresholds and IoU before opening the final labels or running final inference. Use an independently reviewed test export from recording sessions held out of training, synthesis and validation. Verify recording origins and split isolation upstream. The source declaration and annotations fingerprint alone cannot establish that history.

The following can live in the backend-owned handler for the existing synchronous `POST /api/evaluations` once the controller supplies already verified, frozen inputs. Merge `codex/evaluation-engine` into the integration branch. Call this function with `request` from that route, `manifest` and `models` loaded from trusted storage, and `config_by_id`, `variant_by_id`, `class_maps`, and `readiness` from the frozen validation operating points and test policy. Once the call succeeds, save the returned object with the route's existing `storage.save("reports", report)`.

```python
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from backend import schemas
from backend.infrastructure import gpu_client
from backend.evaluation_engine.yolo import (
    YoloLabelImage, build_test_per_model, evaluate_yolo,
    from_inference_result, load_yolo_ground_truth,
)


def run_final_evaluation(request, models, manifest, *, config_by_id, variant_by_id,
                         class_maps, readiness):
    if manifest.dataset_id != request.test_dataset_id or manifest.variant != "test":
        raise ValueError("expected the requested test manifest")
    if not manifest.images or any(
        item.split != "test" or item.synthetic or item.review_status != "reviewed"
        for item in manifest.images
    ):
        raise ValueError("expected a nonempty test roster of reviewed real frames")

    # GX10 owns the dataset files. Obtain original dimensions from reviewed annotations.
    annotation_set = schemas.AnnotationSet.model_validate(gpu_client.post(
        "/datasets/annotations", json={"manifest": manifest.model_dump(mode="json")}
    ))
    if annotation_set.class_names != manifest.class_names:
        raise ValueError("manifest and annotation class orders differ")
    annotations = {entry.frame_id: entry for entry in annotation_set.annotations}
    root = PurePosixPath(manifest.root)

    with TemporaryDirectory() as directory:
        roster = []
        for index, item in enumerate(manifest.images):
            annotation = annotations.get(item.frame_id)
            if (annotation is None or annotation.status != "reviewed" or annotation.synthetic
                    or annotation.session_id != item.session_id):
                raise ValueError(f"{item.frame_id}: missing reviewed real annotation")
            label_path = Path(directory) / f"{index}.txt"
            with gpu_client.stream_file(str(root / item.label)) as response:
                label_path.write_bytes(response.content)
            roster.append(YoloLabelImage(
                item.frame_id, label_path, annotation.width, annotation.height
            ))
        truth = load_yolo_ground_truth(
            roster, dataset_id=manifest.dataset_id, split="test",
            source="independently_labelled_real", class_names=manifest.class_names,
        )

    reports = []
    for model in models:
        config = config_by_id[model.id]
        rows = []
        for item in manifest.images:
            remote = root / item.image
            with gpu_client.stream_file(str(remote)) as response:
                image_bytes = response.content
                mime_type = response.headers.get("content-type", "application/octet-stream")
            result = schemas.InferenceResult.model_validate(gpu_client.post(
                "/infer",
                data={"model": model.model_dump_json(), "conf": config.confidence_threshold},
                files={"image": (remote.name, image_bytes, mime_type)},
                timeout=60,
            ))
            if result.model_id != model.id:
                raise ValueError(f"{item.frame_id}: inference returned a different model ID")
            rows.append(from_inference_result(
                item.frame_id, result.model_dump(), class_map=class_maps.get(model.id)
            ))
        reports.append(evaluate_yolo(
            truth, rows, config, model_id=model.id,
            variant=variant_by_id[model.id], readiness=readiness,
        ))

    return schemas.EvaluationReport(
        request=request, mock=False, per_model=build_test_per_model(
            reports, dataset_id=request.test_dataset_id, model_ids=request.model_ids,
        ),
    )
```

Use the route's existing worker error translation for `GPUWorkerError` and surface adapter input errors as HTTP 422. The example sends the frozen inference confidence threshold to GX10. The existing worker rounds scores and coordinates. Use raw Ultralytics Results on the GX10 if exact matching near score/IoU boundaries matters. For validation sweeps, collect predictions at a floor at or below the lowest candidate threshold.

## Verify and reproduce

```sh
python -m pip install -r backend/evaluation_engine/tests/requirements.txt
python -m backend.evaluation_engine.examples.yolo.generate
python -m pytest -q backend/evaluation_engine/tests
```

The original 34 tests and 20 adapter integration cases passed locally on Python 3.10. The fixtures cover real-format label files, the Ultralytics Results interface and the GX10 payload shape. The real GX10 endpoint still needs a smoke test on reviewed validation frames. Final test execution requires the frozen artifacts, thresholds and independently reviewed held-out export. The backend owner should confirm the nested `per_model` contract before enabling the route.
