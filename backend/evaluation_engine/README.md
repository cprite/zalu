# ZALU Evaluation Engine

Evaluate detection predictions against independently labelled frames. The engine runs on Python 3.10+ with the standard library. Tests use pytest and jsonschema.

```sh
python -m pip install -r backend/evaluation_engine/tests/requirements.txt
python -m pytest -q backend/evaluation_engine/tests
```

## Python API

The checked-in examples use synthetic fixtures. The scores describe those fixtures only.

```python
import json
from pathlib import Path

from backend.evaluation_engine import (
    EvaluationConfig, ReadinessThresholds, build_evaluation_report,
    compare_models, evaluate_detections, load_json,
)

examples = Path("backend/evaluation_engine/examples")
truth = load_json(examples / "ground_truth.synthetic.json")
config = EvaluationConfig(iou_threshold=0.5, confidence_threshold=0.5)
policy = ReadinessThresholds(
    min_precision=0.8, min_recall=0.8, min_f1=0.8,
    min_images=4, min_ground_truth=4, min_class_support={"vest": 2},
)  # Illustrative fixture thresholds.

def score(variant):
    predictions = load_json(examples / f"{variant}.synthetic.json")
    return build_evaluation_report(
        evaluate_detections(truth, predictions, config), readiness=policy,
    )

baseline = score("pretrained_baseline")
fine_tuned = score("fine_tuned")
comparison = compare_models(baseline, fine_tuned)
print(json.dumps(comparison, indent=2, allow_nan=False))
```

## Input contract

| Document | Required top-level fields |
| --- | --- |
| Ground truth | `dataset_id`, `split` (`validation` or `test`), `source` (`independently_labelled_real` or `synthetic_fixture`), unique `class_names`, `images` |
| Predictions | `model_id`, `images`. Optional `variant` is one of `pretrained_baseline`, `fine_tuned`, `real_only`, `real_plus_synthetic`, `unspecified`. |

Each image has an `image_id` and a `boxes` array. Ground truth can also supply `width` and `height` together. Every box has a `class_name` and `xyxy` in original-image pixels with positive area. Prediction boxes also carry `confidence` in [0, 1]. Class names are case-sensitive. Ground truth declares all target classes, including classes with no objects in this split.

An explicit empty ground-truth box list marks a labelled background image. A prediction image with an empty box list records a completed inference with zero detections. Omitted prediction images count as zero detections and increment `images_without_prediction_rows`. Export one row per successfully processed image so the caller can catch an incomplete inference run. Duplicate image IDs, extra prediction image IDs, undeclared ground-truth classes and invalid boxes raise contextual `ValueError`. Valid predictions of undeclared classes count as false positives. All input boxes are validated before confidence filtering.

## Scoring and diagnostics

- Within one image, consider predictions in descending confidence order. Each prediction takes the highest-IoU unassigned ground-truth box of the same class at or above the IoU threshold. Both thresholds are inclusive. Coordinate and class ordering break score ties reproducibly.
- Compute precision as TP / (TP + FP), recall as TP / (TP + FN) and F1 as 2TP / (2TP + FP + FN). Zero denominators produce zero. The summary pools counts over all images and classes before computing metrics. The report includes support (TP + FN) per class.
- After matching, pair residual overlapping boxes of different classes one-to-one by descending IoU for confusion analysis. A wrong-class pair contributes one FP and one FN to scoring. Confusion rows use `null` for background or a missing detection. Diagonal cells show true positives. Off-diagonal class pairs show residual class mismatches. Other FPs and FNs appear in the corresponding `null` row or column.
- The Failure Gallery uses `failures[]`. Each item contains the image ID, available dimensions, all ground-truth boxes and above-threshold predictions, and error entries with zero-based indices into those arrays. A localization error references the closest overlapping ground-truth box below the IoU threshold. A duplicate overlaps an already matched object of the same class. Remaining high-IoU overlaps are `unpaired_overlap`. Standalone FPs without overlap are `background`. Unpaired FNs are `missed`. Error categories are diagnostic and can reference the same object, so use the summary for FP and FN totals.

## Validation, comparison and final test

1. Build independently labelled validation and test sets from separate real recording sessions. Freeze the final test roster and labels before tuning. Use validation predictions to try candidate models and confidence thresholds. Run inference at a floor at or below every candidate threshold to preserve low-confidence candidates.
2. Pass the resulting validation reports to `select_on_validation`. It ranks by micro F1, then precision, then higher confidence threshold, then model ID. Select within each comparison arm if both arms need independent tuning. Choose the model from validation and freeze each operating point, the IoU threshold and the readiness policy before final evaluation.
3. Evaluate frozen artifacts on the same independent test roster. Use `compare_models(a, b)` for descriptive B-minus-A changes, either pretrained baseline versus fine-tuned or real-only versus real-plus-synthetic. The engine checks dataset metadata, a canonical fingerprint of image IDs/dimensions/labels and the IoU threshold. Models may have different frozen confidence thresholds, which are reported. `improved_classes` and `regressed_classes` reflect the sign of each class's F1 change.
4. Readiness uses explicitly configured minimum micro precision, recall, F1, image count, object count and optional per-class support. With thresholds supplied, validation and synthetic reports are `ineligible`. A report with no policy is `not_configured`. `passed` means an eligible report met the configured checks.

## Backend handoff

For YOLO label files, Ultralytics Results, GX10 inference JSON and the FastAPI response envelope, see the [integration handoff](INTEGRATION.md).

For the `kolyan/backend-p0` contract, start from a `DatasetManifest` with `variant="test"` and test-only real sessions. The current Data Engine export can contain auto-labelled frames. Require reviewed annotations, verify their completeness, and confirm the recording sessions are disjoint from training and validation before assembling ground truth. Keep the frame roster from the manifest, including reviewed background frames, and use the corresponding reviewed `AnnotationSet` boxes in pixel coordinates.

Example mapping after the caller has validated the manifest, provenance and inference completion. `reviewed_by_frame` maps frame IDs to shared `Annotation` objects. `inference_by_frame` maps the same IDs to a single model's shared `InferenceResult` objects.

```python
truth = {
    "dataset_id": manifest.dataset_id,
    "split": "test",
    "source": "independently_labelled_real",
    "class_names": manifest.class_names,
    "images": [
        {
            "image_id": item.frame_id,
            "width": annotation.width,
            "height": annotation.height,
            "boxes": [
                {"class_name": box.class_name, "xyxy": box.xyxy}
                for box in annotation.boxes
            ],
        }
        for item in manifest.images
        for annotation in [reviewed_by_frame[item.frame_id]]
    ],
}
predictions = {
    "model_id": model.id,
    "variant": model.variant or "unspecified",  # Supply pretrained_baseline for the baseline arm.
    "images": [
        {
            "image_id": item.frame_id,
            "boxes": [
                {"class_name": detection.cls, "xyxy": detection.box, "confidence": detection.score}
                for detection in inference_by_frame[item.frame_id].detections
            ],
        }
        for item in manifest.images
    ],
}
report = build_evaluation_report(evaluate_detections(truth, predictions, frozen_config))
```

Map deployment labels to the agreed `TaskSpec.class_names` before scoring. Use exact model outputs when available. The current Deployment Engine rounds scores and coordinates, which can affect matches at thresholds.

Proposed API mapping for review with Kolyan: store the standalone report in `backend.schemas.EvaluationReport.per_model[model_id]` within the existing request/report envelope. Keep the endpoint's existing envelope metadata and request IDs. The current FastAPI endpoint looks for an `evaluate(req, models, test)` hook. That adapter requires prediction execution, reviewed annotation retrieval and coordination on the response contract. Merge the engine branch into the integration branch and agree on the adapter before switching the endpoint to real reports.

## Frontend examples

- [EvaluationReport JSON Schema](evaluation_report.schema.json) defines the version 1.0 contract.
- [Baseline report with Failure Gallery data](examples/baseline_report.synthetic.json) and [fine-tuned report](examples/fine_tuned_report.synthetic.json) show the Model Report Card fields.
- [Baseline versus fine-tuned](examples/baseline_vs_fine_tuned.synthetic.json) and [real-only versus real-plus-synthetic](examples/real_vs_real_plus_synthetic.synthetic.json) show comparison output.
- Regenerate the four output examples with `python -m backend.evaluation_engine.examples.generate`.

Join `image_id` to a backend-authorized image URL by dataset and frame ID. The report supplies indices and pixel coordinates for overlays. It does not supply image URLs.

## Limitations

The engine measures axis-aligned boxes at a single operating point. It does not compute AP/mAP, confidence intervals, temporal/event metrics or latency. Confidence-ranked matching is greedy and can differ from maximum-cardinality matching on crowded images. Pairwise matching takes memory proportional to the product of the prediction and ground-truth counts for an image. The gallery includes all failed images, so the serving layer may need pagination for large reports.

Dataset provenance, label completeness and session isolation are the caller's responsibility. The fingerprint covers the annotated roster and coordinates, not the image bytes. The engine cannot establish that a threshold was frozen before test evaluation. The existing backend hook, remote inference flow and endpoints require integration testing after real predictions and reviewed annotations can be supplied.
