"""YOLO input adapters for the shared evaluator's pixel-coordinate Box type."""

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

from backend.evaluation_engine.matching import Box


# Data Engine serializes normalized coordinates to six decimal places.
_EDGE_TOLERANCE = 1e-6


def _size(size: tuple[int, int]) -> None:
    if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
        raise ValueError(f"expected positive integer (width, height), got {size!r}")


def _pixel_box(image: str, label: str, coords, size: tuple[int, int], score=None) -> Box:
    w, h = size
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{image}: expected a nonempty class name")
    if len(coords) != 4 or any(
        isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in coords
    ):
        raise ValueError(f"{image}: expected four finite xyxy coordinates")
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"{image}: invalid xyxy box {coords!r} for image size {size}")
    return Box.of(image, label, coords, score)


def parse_yolo_labels(
    image: str, text: str, size: tuple[int, int], class_names: Sequence[str], *,
    label_path: str | None = None,
) -> list[Box]:
    """Convert five-column normalized YOLO detection labels to pixel boxes."""
    _size(size)
    w, h = size
    boxes = []
    for line, row in enumerate(text.splitlines(), 1):
        if not row.strip():
            continue
        context = f"{label_path or image}:{line}"
        try:
            cls, cx, cy, bw, bh = row.split()
            class_id = int(cls)
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        except ValueError as exc:
            raise ValueError(f"{context}: expected class_id cx cy width height") from exc
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"{context}: unknown dataset class ID {class_id}")
        if not all(0 <= v <= 1 for v in (cx, cy, bw, bh)) or not bw or not bh:
            raise ValueError(f"{context}: invalid normalized box")
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if min(x1, y1) < -_EDGE_TOLERANCE or max(x2, y2) > 1 + _EDGE_TOLERANCE:
            raise ValueError(f"{context}: box extends outside the image")
        coords = [max(0, x1) * w, max(0, y1) * h, min(1, x2) * w, min(1, y2) * h]
        boxes.append(_pixel_box(image, class_names[class_id], coords, size))
    return boxes


def from_inference_result(
    image: str, result, *, size: tuple[int, int], class_map: Mapping[str, str] | None = None,
    model_id: str | None = None,
) -> list[Box]:
    """Convert GX10 /infer JSON or a shared InferenceResult on the specified original image."""
    packet = result if isinstance(result, Mapping) else result.model_dump()
    packet_size = (packet.get("width"), packet.get("height"))
    _size(size)
    _size(packet_size)
    if packet_size != size:
        raise ValueError(f"{image}: inference dimensions {packet_size} differ from original image {size}")
    if model_id is not None and packet.get("model_id") != model_id:
        raise ValueError(f"{image}: inference returned a different model ID")
    detections = packet.get("detections")
    if not isinstance(detections, list):
        raise ValueError(f"{image}: expected a detections array")
    class_map = class_map or {}
    boxes = []
    for detection in detections:
        try:
            score, label, coords = detection["score"], detection["cls"], detection["box"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{image}: expected detection cls, box and score") from exc
        if isinstance(score, bool) or not isinstance(score, Real) or not 0 <= score <= 1:
            raise ValueError(f"{image}: invalid detection confidence")
        if not isinstance(label, str) or not isinstance(coords, (list, tuple)):
            raise ValueError(f"{image}: invalid detection label or box")
        boxes.append(_pixel_box(image, class_map.get(label, label), coords, size, float(score)))
    return boxes


def from_ultralytics_result(
    image: str, result, *, size: tuple[int, int], class_map: Mapping[str, str] | None = None,
) -> list[Box]:
    """Convert one raw Ultralytics detection Results object without importing Ultralytics."""
    if result.boxes is None:
        raise ValueError(f"{image}: expected an Ultralytics detection result")
    h, w = result.orig_shape
    boxes = result.boxes
    coords, scores, ids = boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    if len(coords) != len(scores) or len(coords) != len(ids):
        raise ValueError(f"{image}: inconsistent Ultralytics detection array lengths")
    detections = []
    for xyxy, score, raw_id in zip(coords, scores, ids):
        if (isinstance(raw_id, bool) or not isinstance(raw_id, Real)
                or not isfinite(raw_id) or raw_id < 0 or raw_id != int(raw_id)):
            raise ValueError(f"{image}: invalid model class ID {raw_id!r}")
        try:
            label = result.names[int(raw_id)]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"{image}: unknown model class ID {raw_id}") from exc
        detections.append({"cls": label, "box": xyxy, "score": score})
    return from_inference_result(
        image, {"width": w, "height": h, "detections": detections}, size=size, class_map=class_map,
    )
