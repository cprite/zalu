"""Input validation and JSON loading for detection evaluation."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.25

    def __post_init__(self):
        probability(self.iou_threshold, "iou_threshold")
        probability(self.confidence_threshold, "confidence_threshold")
        if self.iou_threshold == 0:
            raise ValueError("iou_threshold must be greater than zero")


def load_json(path: str | Path) -> dict:
    """Load a UTF-8 JSON object with file context on errors."""
    path = Path(path)
    try:
        return object_value(json.loads(path.read_text(encoding="utf-8")), str(path))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def object_value(value, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def array(value, path):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array")
    return value


def name(value, path):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{path} must be a nonempty, trimmed string")
    return value


def number(value, path):
    if type(value) not in (int, float) or not -1e100 <= value <= 1e100:
        raise ValueError(f"{path} must be a finite number within ±1e100")
    return float(value)


def probability(value, path):
    value = number(value, path)
    if not 0 <= value <= 1:
        raise ValueError(f"{path} must be between zero and one")
    return value


def positive_int(value, path):
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _box(value, path, prediction, size=None):
    row = object_value(value, path)
    label = name(row.get("class_name"), f"{path}.class_name")
    coords = array(row.get("xyxy"), f"{path}.xyxy")
    if len(coords) != 4:
        raise ValueError(f"{path}.xyxy must contain four coordinates")
    x1, y1, x2, y2 = (number(v, f"{path}.xyxy") + 0.0 for v in coords)
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        raise ValueError(f"{path}.xyxy must have nonnegative origins and positive area")
    if size and (x2 > size[0] or y2 > size[1]):
        raise ValueError(f"{path}.xyxy exceeds image bounds")
    area = (x2 - x1) * (y2 - y1)
    if not isfinite(area) or area == 0:
        raise ValueError(f"{path}.xyxy has an unrepresentable area")
    box = {"class_name": label, "xyxy": [x1, y1, x2, y2]}
    if prediction:
        box["confidence"] = probability(row.get("confidence"), f"{path}.confidence")
    return box


def _images(rows, ground_truth=None):
    prediction = ground_truth is not None
    kind = "predictions" if prediction else "ground_truth"
    images = {}
    for index, value in enumerate(array(rows, f"{kind}.images")):
        path = f"{kind}.images[{index}]"
        row = object_value(value, path)
        image_id = name(row.get("image_id"), f"{path}.image_id")
        path += f" ({image_id!r})"
        if image_id in images:
            raise ValueError(f"{path}: duplicate image id")
        if prediction and image_id not in ground_truth:
            raise ValueError(f"{path}: image is absent from ground truth")
        image = {}
        if prediction:
            reference = ground_truth[image_id]
            size = (reference["width"], reference["height"]) if "width" in reference else None
        elif "width" in row or "height" in row:
            size = (positive_int(row.get("width"), f"{path}.width"),
                    positive_int(row.get("height"), f"{path}.height"))
            image.update(width=size[0], height=size[1])
        else:
            size = None
        image["boxes"] = [
            _box(box, f"{path}.boxes[{i}]", prediction, size)
            for i, box in enumerate(array(row.get("boxes"), f"{path}.boxes"))
        ]
        images[image_id] = image
    return images


def parse_inputs(ground_truth, predictions):
    truth = object_value(ground_truth, "ground_truth")
    pred = object_value(predictions, "predictions")
    dataset_id = name(truth.get("dataset_id"), "ground_truth.dataset_id")
    split = truth.get("split")
    if split not in ("validation", "test"):
        raise ValueError("ground_truth.split must be validation or test")
    source = truth.get("source")
    if source not in ("independently_labelled_real", "synthetic_fixture"):
        raise ValueError("ground_truth.source must identify independently labelled real data or a synthetic fixture")
    classes = [name(c, "ground_truth.class_names entry")
               for c in array(truth.get("class_names"), "ground_truth.class_names")]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("ground_truth.class_names must contain unique classes")
    images = _images(truth.get("images"))
    allowed = set(classes)
    for image_id, image in images.items():
        for box in image["boxes"]:
            if box["class_name"] not in allowed:
                raise ValueError(f"ground truth image {image_id!r}: undeclared class {box['class_name']!r}")
    model_id = name(pred.get("model_id"), "predictions.model_id")
    variant = pred.get("variant", "unspecified")
    if variant not in ("pretrained_baseline", "fine_tuned", "real_only", "real_plus_synthetic", "unspecified"):
        raise ValueError("predictions.variant is unknown")
    dataset = {"id": dataset_id, "split": split, "source": source, "class_names": sorted(classes)}
    return dataset, images, {"id": model_id, "variant": variant}, _images(pred.get("images"), images)
