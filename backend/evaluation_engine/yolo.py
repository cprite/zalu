"""Convert YOLO labels and inference results to the evaluation input contract."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .detection import evaluate_detections
from .inputs import EvaluationConfig, name, number, positive_int, probability
from .reporting import EvaluationReport, ReadinessThresholds, build_evaluation_report


# Data Engine exports normalized labels with six decimal places.
_LABEL_EDGE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class YoloLabelImage:
    image_id: str
    label_path: str | Path
    width: int
    height: int


def _label_boxes(image: YoloLabelImage, class_names: Sequence[str]) -> list[dict]:
    path = Path(image.label_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{image.image_id}: cannot read YOLO labels at {path}") from exc

    boxes = []
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        context = f"{path}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"{context}: expected class_id cx cy width height")
        try:
            class_id = int(fields[0])
            coordinates = [float(field) for field in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{context}: invalid YOLO class ID or coordinates") from exc
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"{context}: unknown ground-truth class ID {class_id}")
        cx, cy, w, h = (probability(value, context) for value in coordinates)
        if not w or not h:
            raise ValueError(f"{context}: box width and height must be positive")
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if min(x1, y1) < -_LABEL_EDGE_TOLERANCE or max(x2, y2) > 1 + _LABEL_EDGE_TOLERANCE:
            raise ValueError(f"{context}: box extends outside the image")
        boxes.append({
            "class_name": class_names[class_id],
            "xyxy": [max(0, x1) * image.width, max(0, y1) * image.height,
                     min(1, x2) * image.width, min(1, y2) * image.height],
        })
    return boxes


def load_yolo_ground_truth(
    images: Iterable[YoloLabelImage], *, dataset_id: str, split: str, source: str,
    class_names: Sequence[str],
) -> dict:
    """Read normalized YOLO detection labels against an authoritative image roster."""
    rows = []
    seen = set()
    for image in images:
        image_id = name(image.image_id, "image_id")
        if image_id in seen:
            raise ValueError(f"duplicate ground-truth image ID {image_id!r}")
        seen.add(image_id)
        positive_int(image.width, f"{image_id}.width")
        positive_int(image.height, f"{image_id}.height")
        rows.append({
            "image_id": image_id, "width": image.width, "height": image.height,
            "boxes": _label_boxes(image, class_names),
        })
    return {"dataset_id": dataset_id, "split": split, "source": source,
            "class_names": list(class_names), "images": rows}


def from_ultralytics_result(
    image_id: str, result, *, class_map: Mapping[str, str] | None = None,
) -> dict:
    """Adapt one Ultralytics detection Results object without importing Ultralytics."""
    if result.boxes is None:
        raise ValueError(f"{image_id}: expected an Ultralytics detection result with boxes")
    height, width = result.orig_shape
    boxes = result.boxes
    coords, scores, ids = boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    if len(coords) != len(scores) or len(coords) != len(ids):
        raise ValueError(f"{image_id}: inconsistent Ultralytics detection array lengths")
    detections = []
    for xyxy, score, raw_id in zip(coords, scores, ids):
        value = number(raw_id, f"{image_id}.class_id")
        class_id = int(value)
        if value != class_id or class_id < 0:
            raise ValueError(f"{image_id}: invalid prediction class ID {raw_id}")
        try:
            label = result.names[class_id]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"{image_id}: no model name for prediction class ID {class_id}") from exc
        detections.append({"cls": label, "box": xyxy, "score": score})
    return from_inference_result(
        image_id, {"width": width, "height": height, "detections": detections},
        class_map=class_map,
    )


def from_inference_result(
    image_id: str, result: Mapping, *, class_map: Mapping[str, str] | None = None,
) -> dict:
    """Adapt a GX10 /infer JSON response or InferenceResult.model_dump()."""
    class_map = {} if class_map is None else class_map
    boxes = []
    for detection in result["detections"]:
        label = name(detection["cls"], f"{image_id}.cls")
        boxes.append({
            "class_name": name(class_map.get(label, label), f"{image_id}.mapped_class_name"),
            "xyxy": list(detection["box"]), "confidence": detection["score"],
        })
    return {
        "image_id": name(image_id, "image_id"),
        "width": positive_int(result["width"], f"{image_id}.width"),
        "height": positive_int(result["height"], f"{image_id}.height"),
        "boxes": boxes,
    }


def evaluate_yolo(
    ground_truth: dict, prediction_rows: Iterable[dict], config: EvaluationConfig, *,
    model_id: str, variant: str = "unspecified", readiness: ReadinessThresholds | None = None,
) -> EvaluationReport:
    """Report on a complete YOLO inference run, validating the image roster and dimensions."""
    dimensions = {row["image_id"]: (row["width"], row["height"]) for row in ground_truth["images"]}
    if not dimensions:
        raise ValueError("YOLO evaluation requires at least one image")
    rows = list(prediction_rows)
    seen = set()
    for row in rows:
        image_id = row["image_id"]
        if image_id in seen:
            raise ValueError(f"duplicate prediction image ID {image_id!r}")
        seen.add(image_id)
        if image_id not in dimensions:
            raise ValueError(f"{image_id}: prediction image absent from YOLO ground truth")
        if (row["width"], row["height"]) != dimensions[image_id]:
            raise ValueError(f"{image_id}: prediction dimensions differ from ground truth")
    missing = dimensions.keys() - seen
    if missing:
        raise ValueError(f"missing inference rows ({len(missing)}), first: {min(missing)!r}")
    predictions = {"model_id": model_id, "variant": variant, "images": rows}
    return build_evaluation_report(
        evaluate_detections(ground_truth, predictions, config), readiness=readiness,
    )


def build_test_per_model(
    reports: Iterable[EvaluationReport], *, dataset_id: str, model_ids: Sequence[str],
) -> dict[str, EvaluationReport]:
    """Build the API's per_model field for a real final test request."""
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("expected unique requested model IDs")
    per_model = {}
    reference = None
    for report in reports:
        dataset = report["dataset"]
        if (dataset["id"] != dataset_id or dataset["split"] != "test"
                or dataset["source"] != "independently_labelled_real"):
            raise ValueError("final test API requires reports for the requested independently labelled real test dataset")
        if report["summary"]["images_without_prediction_rows"]:
            raise ValueError("final test API requires complete inference rows")
        comparable = (dataset, report["config"]["iou_threshold"])
        if reference is not None and reference != comparable:
            raise ValueError("final test models must share the ground truth and IoU threshold")
        reference = comparable
        model_id = report["model"]["id"]
        if model_id in per_model:
            raise ValueError(f"duplicate report for model {model_id!r}")
        per_model[model_id] = report
    if per_model.keys() != set(model_ids):
        raise ValueError("report model IDs differ from requested model IDs")
    return {model_id: per_model[model_id] for model_id in model_ids}
