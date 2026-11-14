"""Confidence-ranked, class-aware matching of axis-aligned detections."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json

from .inputs import EvaluationConfig, parse_inputs


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return min(1.0, intersection / union)


def _box_key(box):
    return box["class_name"], box["xyxy"]


def _match(truth, pred, threshold):
    overlaps = [[_iou(p, gt) for gt in truth] for p in pred]
    remaining = set(range(len(truth)))
    matches = {}
    for p, prediction in enumerate(pred):
        candidates = (
            g for g in remaining
            if truth[g]["class_name"] == prediction["class_name"] and overlaps[p][g] >= threshold
        )
        g = max(candidates, key=lambda g: (overlaps[p][g], -g), default=None)
        if g is not None:
            matches[p] = g
            remaining.remove(g)

    unmatched_predictions = set(range(len(pred))) - matches.keys()
    # Pair residual wrong classes for diagnostics after locking all true positives.
    candidates = sorted(
        (-overlaps[p][g], p, g)
        for p in unmatched_predictions
        for g in remaining
        if overlaps[p][g] >= threshold and pred[p]["class_name"] != truth[g]["class_name"]
    )
    errors = []
    for neg_iou, p, g in candidates:
        if p in unmatched_predictions and g in remaining:
            errors.append(_error("class_mismatch", g, p, -neg_iou))
            unmatched_predictions.remove(p)
            remaining.remove(g)

    for p in sorted(unmatched_predictions):
        same_class = [g for g in matches.values() if truth[g]["class_name"] == pred[p]["class_name"]]
        duplicate = max(same_class, key=lambda g: (overlaps[p][g], -g), default=None)
        if duplicate is not None and overlaps[p][duplicate] >= threshold:
            errors.append(_error("duplicate", duplicate, p, overlaps[p][duplicate]))
            continue
        closest = max(range(len(truth)), key=lambda g: (overlaps[p][g], -g), default=None)
        overlap = overlaps[p][closest] if closest is not None else 0
        if overlap > 0:
            kind = "localization" if overlap < threshold else "unpaired_overlap"
            errors.append(_error(kind, closest, p, overlap))
        else:
            errors.append(_error("background", None, p, None))
    errors.extend(_error("missed", g, None, None) for g in sorted(remaining))
    return matches, errors


def _error(kind, gt, pred, iou):
    return {"type": kind, "ground_truth_index": gt, "prediction_index": pred, "iou": iou}


def _metrics(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def evaluate_detections(ground_truth, predictions, config: EvaluationConfig) -> dict:
    """Evaluate one model on the supplied image roster and return JSON-ready results."""
    if not isinstance(config, EvaluationConfig):
        raise ValueError("config must be an EvaluationConfig")
    dataset, truth_images, model, prediction_images = parse_inputs(ground_truth, predictions)
    tp, gt_counts, pred_counts = Counter(), Counter(), Counter()
    confusion, error_counts = Counter(), Counter()
    failures = []
    filtered = background = 0

    for image_id, image in sorted(truth_images.items()):
        truth = sorted(image["boxes"], key=_box_key)
        image["boxes"] = truth
        raw = prediction_images.get(image_id, {}).get("boxes", [])
        pred = sorted(
            (box for box in raw if box["confidence"] >= config.confidence_threshold),
            key=lambda box: (-box["confidence"], *_box_key(box)),
        )
        filtered += len(raw) - len(pred)
        background += not truth
        gt_counts.update(box["class_name"] for box in truth)
        pred_counts.update(box["class_name"] for box in pred)

        matches, errors = _match(truth, pred, config.iou_threshold)
        for g in matches.values():
            label = truth[g]["class_name"]
            tp[label] += 1
            confusion[label, label] += 1
        for error in errors:
            g, p = error["ground_truth_index"], error["prediction_index"]
            actual = truth[g]["class_name"] if g is not None else None
            predicted = pred[p]["class_name"] if p is not None else None
            if error["type"] in ("class_mismatch", "missed"):
                confusion[actual, predicted] += 1
            else:
                confusion[None, predicted] += 1
            error_counts[error["type"]] += 1
        if errors:
            failure = {
                "image_id": image_id, "ground_truth": truth, "predictions": pred, "errors": errors
            }
            if "width" in image:
                failure.update(width=image["width"], height=image["height"])
            failures.append(failure)

    classes = sorted(set(dataset["class_names"]) | pred_counts.keys())
    per_class = {
        c: _metrics(tp[c], pred_counts[c] - tp[c], gt_counts[c] - tp[c]) for c in classes
    }
    total_tp = tp.total()
    canonical = json.dumps(truth_images, sort_keys=True, separators=(",", ":"), allow_nan=False)
    dataset.update(
        fingerprint=sha256(canonical.encode("utf-8")).hexdigest(),
        image_count=len(truth_images), background_image_count=background,
        ground_truth_count=gt_counts.total(),
    )
    return {
        "dataset": dataset, "model": model, "config": asdict(config),
        "summary": {
            **_metrics(total_tp, pred_counts.total() - total_tp, gt_counts.total() - total_tp),
            "predictions_below_threshold": filtered,
            "images_without_prediction_rows": len(truth_images.keys() - prediction_images.keys()),
            "failure_image_count": len(failures),
        },
        "per_class": per_class,
        "confusion": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in sorted(
                confusion.items(), key=lambda item: tuple(label or "" for label in item[0])
            )
        ],
        "error_counts": dict(sorted(error_counts.items())),
        "failures": failures,
    }
