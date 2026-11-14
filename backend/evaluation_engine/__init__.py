"""ZALU's standalone object detection evaluation API."""

from .detection import evaluate_detections
from .inputs import EvaluationConfig, load_json
from .reporting import (
    EvaluationReport,
    ReadinessThresholds,
    build_evaluation_report,
    compare_models,
    select_on_validation,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationReport",
    "ReadinessThresholds",
    "build_evaluation_report",
    "compare_models",
    "evaluate_detections",
    "load_json",
    "select_on_validation",
]
