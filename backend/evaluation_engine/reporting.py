"""Versioned reports, readiness checks, and paired model comparisons."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict

from .inputs import name, positive_int, probability


class EvaluationReport(TypedDict):
    schema_version: Literal["1.0"]
    task_type: Literal["object_detection"]
    dataset: dict
    model: dict
    config: dict
    summary: dict
    per_class: dict
    confusion: list[dict]
    error_counts: dict
    failures: list[dict]
    readiness: dict


@dataclass(frozen=True)
class ReadinessThresholds:
    min_precision: float
    min_recall: float
    min_f1: float
    min_images: int
    min_ground_truth: int
    min_class_support: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for metric in ("precision", "recall", "f1"):
            key = f"min_{metric}"
            probability(getattr(self, key), key)
        positive_int(self.min_images, "min_images")
        positive_int(self.min_ground_truth, "min_ground_truth")
        if not isinstance(self.min_class_support, dict):
            raise ValueError("min_class_support must be a dictionary")
        for label, minimum in self.min_class_support.items():
            name(label, "min_class_support class")
            positive_int(minimum, f"min_class_support[{label!r}]")


def _readiness(results, thresholds):
    if thresholds is None:
        return {"status": "not_configured", "thresholds": None, "checks": []}
    if not isinstance(thresholds, ReadinessThresholds):
        raise ValueError("readiness must be ReadinessThresholds")
    dataset, summary = results["dataset"], results["summary"]
    checks = []

    def add(metric, observed, minimum, class_name=None):
        checks.append({
            "metric": metric, "class_name": class_name, "actual": observed,
            "minimum": minimum, "passed": observed >= minimum,
        })

    add("image_count", dataset["image_count"], thresholds.min_images)
    add("ground_truth_count", dataset["ground_truth_count"], thresholds.min_ground_truth)
    for metric in ("precision", "recall", "f1"):
        add(metric, summary[metric], getattr(thresholds, f"min_{metric}"))
    for label, minimum in sorted(thresholds.min_class_support.items()):
        if label not in dataset["class_names"]:
            raise ValueError(f"readiness: undeclared class {label!r}")
        add("support", results["per_class"][label]["support"], minimum, label)
    if dataset["source"] != "independently_labelled_real" or dataset["split"] != "test":
        status = "ineligible"
    else:
        status = "passed" if all(check["passed"] for check in checks) else "failed"
    return {"status": status, "thresholds": asdict(thresholds), "checks": checks}


def build_evaluation_report(results, *, readiness: ReadinessThresholds | None = None) -> EvaluationReport:
    """Build a versioned JSON-ready report from evaluate_detections output."""
    return {
        "schema_version": "1.0", "task_type": "object_detection",
        **deepcopy(results), "readiness": _readiness(results, readiness),
    }


def _check_compatibility(report_a, report_b):
    for report in (report_a, report_b):
        if report["schema_version"] != "1.0" or report["task_type"] != "object_detection":
            raise ValueError("expected an object detection report with schema_version 1.0")
    if report_a["dataset"] != report_b["dataset"]:
        raise ValueError("reports must use the same ground truth, image roster, taxonomy and split")
    if report_a["config"]["iou_threshold"] != report_b["config"]["iou_threshold"]:
        raise ValueError("reports must use the same IoU threshold")


def _delta(a, b):
    return {metric: b[metric] - a[metric] for metric in ("tp", "fp", "fn", "precision", "recall", "f1")}


def compare_models(report_a: EvaluationReport, report_b: EvaluationReport) -> dict:
    """Describe metric changes from model A to model B on identical ground truth."""
    _check_compatibility(report_a, report_b)
    delta = {}
    a_classes, b_classes = report_a["per_class"], report_b["per_class"]
    zero = dict.fromkeys(("tp", "fp", "fn", "precision", "recall", "f1"), 0)
    for label in sorted(a_classes.keys() | b_classes.keys()):
        delta[label] = _delta(a_classes.get(label, zero), b_classes.get(label, zero))
    return {
        "schema_version": "1.0",
        "dataset": deepcopy(report_a["dataset"]),
        "models": {
            key: {
                **report["model"], "confidence_threshold": report["config"]["confidence_threshold"],
                "readiness_status": report["readiness"]["status"],
            }
            for key, report in (("a", report_a), ("b", report_b))
        },
        "iou_threshold": report_a["config"]["iou_threshold"],
        "summary_delta": _delta(report_a["summary"], report_b["summary"]),
        "per_class_delta": delta,
        "improved_classes": [label for label, row in delta.items() if row["f1"] > 0],
        "regressed_classes": [label for label, row in delta.items() if row["f1"] < 0],
        "failures": {"a": deepcopy(report_a["failures"]), "b": deepcopy(report_b["failures"])},
    }


def select_on_validation(reports) -> EvaluationReport:
    """Choose a model and threshold among validation candidates by micro F1."""
    candidates = list(reports)
    if not candidates:
        raise ValueError("at least one validation report is required")
    first = candidates[0]
    for report in candidates:
        _check_compatibility(first, report)
        if report["dataset"]["split"] != "validation":
            raise ValueError("model selection and threshold tuning require validation reports")
        if not report["dataset"]["ground_truth_count"]:
            raise ValueError("validation needs at least one ground-truth box")
    return min(candidates, key=lambda report: (
        -report["summary"]["f1"],
        -report["summary"]["precision"],
        -report["config"]["confidence_threshold"],
        report["model"]["id"],
    ))
