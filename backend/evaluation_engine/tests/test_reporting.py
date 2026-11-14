"""Validate reports, examples, readiness and comparison safety."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
import pytest

from backend.evaluation_engine import (
    EvaluationConfig, ReadinessThresholds, build_evaluation_report, compare_models,
    evaluate_detections, load_json, select_on_validation,
)


ENGINE = Path(__file__).resolve().parents[1]
EXAMPLES = ENGINE / "examples"
READINESS = ReadinessThresholds(0.8, 0.8, 0.8, 4, 4, {"person": 2, "vest": 2})


def report(variant, *, split="test", threshold=0.5, iou=0.5, truth=None):
    if truth is None:
        truth = load_json(EXAMPLES / "ground_truth.synthetic.json")
    truth = {**truth, "split": split}
    predictions = load_json(EXAMPLES / f"{variant}.synthetic.json")
    return build_evaluation_report(
        evaluate_detections(truth, predictions, EvaluationConfig(iou, threshold)),
        readiness=READINESS,
    )


def test_reports_and_examples_conform_to_schema():
    schema = load_json(ENGINE / "evaluation_report.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for variant, file in (("pretrained_baseline", "baseline_report"), ("fine_tuned", "fine_tuned_report")):
        expected = load_json(EXAMPLES / f"{file}.synthetic.json")
        actual = json.loads(json.dumps(report(variant), allow_nan=False))
        validator.validate(actual)
        assert actual == expected
    del actual["summary"]["fn"]
    with pytest.raises(ValidationError):
        validator.validate(actual)


@pytest.mark.parametrize("variant_a,variant_b,example", [
    ("pretrained_baseline", "fine_tuned", "baseline_vs_fine_tuned"),
    ("real_only", "real_plus_synthetic", "real_vs_real_plus_synthetic"),
])
def test_model_comparisons(variant_a, variant_b, example):
    a, b = report(variant_a), report(variant_b, threshold=0.5)
    compared = compare_models(a, b)
    assert compared == load_json(EXAMPLES / f"{example}.synthetic.json")
    assert compared["summary_delta"]["f1"] == b["summary"]["f1"] - a["summary"]["f1"]
    assert compared["failures"] == {"a": a["failures"], "b": b["failures"]}
    assert compared["models"]["b"]["variant"] == variant_b
    if variant_b == "real_plus_synthetic":
        assert compared["improved_classes"] == ["vest"]
        assert compared["regressed_classes"] == ["person"]
        assert compared["summary_delta"]["f1"] == 0


def test_comparison_requires_same_dataset_revision_and_iou():
    a = report("pretrained_baseline")
    changed = load_json(EXAMPLES / "ground_truth.synthetic.json")
    changed["images"][0]["boxes"][0]["xyxy"][2] += 1
    with pytest.raises(ValueError, match="same ground truth"):
        compare_models(a, report("fine_tuned", truth=changed))
    with pytest.raises(ValueError, match="IoU"):
        compare_models(a, report("fine_tuned", iou=0.6))
    assert compare_models(a, report("fine_tuned", threshold=0.7))["models"]["b"]["confidence_threshold"] == 0.7


def test_validation_selection_and_threshold_tuning():
    low = report("pretrained_baseline", split="validation", threshold=0.49)
    high = report("pretrained_baseline", split="validation", threshold=0.5)
    assert select_on_validation([high, low]) is low
    best = report("fine_tuned", split="validation")
    assert select_on_validation([high, best, low]) is best
    with pytest.raises(ValueError, match="validation"):
        select_on_validation([report("fine_tuned")])
    with pytest.raises(ValueError, match="at least one"):
        select_on_validation([])


def test_readiness_states():
    synthetic = report("fine_tuned")
    assert synthetic["readiness"]["status"] == "ineligible"
    assert all(check["passed"] for check in synthetic["readiness"]["checks"])
    # Mock only the provenance declaration to exercise the state machine.
    truth = load_json(EXAMPLES / "ground_truth.synthetic.json")
    truth["source"] = "independently_labelled_real"
    assert report("fine_tuned", truth=truth)["readiness"]["status"] == "passed"
    assert report("fine_tuned", truth=truth, split="validation")["readiness"]["status"] == "ineligible"
    assert report("pretrained_baseline", truth=truth)["readiness"]["status"] == "failed"
    result = evaluate_detections(
        truth, load_json(EXAMPLES / "fine_tuned.synthetic.json"), EvaluationConfig(),
    )
    gate = ReadinessThresholds(0, 0, 0, 1, 1, {"vest": 3})
    failed = build_evaluation_report(result, readiness=gate)
    assert failed["readiness"]["status"] == "failed"
    assert failed["readiness"]["checks"][-1] == {
        "metric": "support", "actual": 2, "minimum": 3, "class_name": "vest", "passed": False,
    }
    unconfigured = build_evaluation_report(result)
    assert unconfigured["readiness"] == {"status": "not_configured", "thresholds": None, "checks": []}
    original = deepcopy(unconfigured)
    result["failures"].append({})
    assert unconfigured == original
