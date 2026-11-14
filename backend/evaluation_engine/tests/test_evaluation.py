"""Exact metrics and edge cases on synthetic fixtures."""

from copy import deepcopy
from pathlib import Path

import pytest

from backend.evaluation_engine import EvaluationConfig, evaluate_detections, load_json


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def fixture_data():
    return (
        load_json(EXAMPLES / "ground_truth.synthetic.json"),
        load_json(EXAMPLES / "pretrained_baseline.synthetic.json"),
    )


def test_exact_baseline_metrics_and_confusion(fixture_data):
    result = evaluate_detections(*fixture_data, EvaluationConfig(0.5, 0.5))
    assert result["summary"] == {
        "tp": 1, "fp": 4, "fn": 3, "support": 4,
        "precision": 1 / 5, "recall": 1 / 4, "f1": 2 / 9,
        "predictions_below_threshold": 1, "images_without_prediction_rows": 0,
        "failure_image_count": 4,
    }
    assert result["per_class"] == {
        "person": {"tp": 1, "fp": 2, "fn": 1, "support": 2,
                   "precision": 1 / 3, "recall": 1 / 2, "f1": 2 / 5},
        "vest": {"tp": 0, "fp": 1, "fn": 2, "support": 2,
                 "precision": 0, "recall": 0, "f1": 0},
        "unknown": {"tp": 0, "fp": 1, "fn": 0, "support": 0,
                    "precision": 0, "recall": 0, "f1": 0},
    }
    assert {
        (row["actual"], row["predicted"]): row["count"] for row in result["confusion"]
    } == {
        ("person", "person"): 1, ("vest", "person"): 1, ("vest", None): 1,
        ("person", None): 1, (None, "unknown"): 1, (None, "person"): 1, (None, "vest"): 1,
    }
    assert result["error_counts"] == {
        "class_mismatch": 1, "duplicate": 1, "localization": 1, "background": 1, "missed": 2,
    }
    failures = result["failures"]
    error = failures[0]["errors"][0]
    assert error["type"] == "class_mismatch"
    assert failures[0]["ground_truth"][error["ground_truth_index"]]["class_name"] == "vest"
    assert failures[0]["predictions"][error["prediction_index"]]["class_name"] == "person"
    assert failures[0]["image_id"] == "synthetic-01"
    assert failures[0]["width"] == 40
    assert failures[1]["errors"][0]["iou"] == 0.4
    assert result["dataset"]["background_image_count"] == 1


def test_inclusive_thresholds_and_class_priority(fixture_data):
    truth, pred = fixture_data
    # Higher-scoring wrong label must leave the box available to the correct class.
    pred["images"][0]["boxes"].append({
        "class_name": "vest", "xyxy": [20, 0, 30, 10], "confidence": 0.49
    })
    result = evaluate_detections(truth, pred, EvaluationConfig(0.4, 0.49))
    assert result["summary"]["tp"] == 4
    assert result["summary"]["fp"] == 3
    assert result["summary"]["fn"] == 0
    assert result["error_counts"]["unpaired_overlap"] == 1
    assert "class_mismatch" not in result["error_counts"]


def test_confidence_rank_and_one_to_one_on_two_objects(fixture_data):
    truth, pred = fixture_data
    truth["images"] = [{"image_id": "crowd", "boxes": [
        {"class_name": "person", "xyxy": [0, 0, 10, 10]},
        {"class_name": "person", "xyxy": [4, 0, 14, 10]},
    ]}]
    pred["images"] = [{"image_id": "crowd", "boxes": [
        {"class_name": "person", "xyxy": [0, 0, 10, 10], "confidence": 0.8},
        {"class_name": "person", "xyxy": [1, 0, 11, 10], "confidence": 0.9},
    ]}]
    result = evaluate_detections(truth, pred, EvaluationConfig())
    assert result["summary"]["tp"] == result["summary"]["fp"] == result["summary"]["fn"] == 1
    assert result["error_counts"] == {"duplicate": 1, "missed": 1}
    assert result["failures"][0]["errors"][0]["prediction_index"] == 1


def test_missing_rows_and_empty_roster(fixture_data):
    truth, pred = fixture_data
    pred["images"] = []
    empty = evaluate_detections(truth, pred, EvaluationConfig())
    assert empty["summary"]["fn"] == 4
    assert empty["summary"]["images_without_prediction_rows"] == 4
    assert empty["summary"]["precision"] == empty["summary"]["recall"] == empty["summary"]["f1"] == 0
    truth["images"] = []
    result = evaluate_detections(truth, pred, EvaluationConfig())
    assert result["summary"]["support"] == 0
    assert result["failures"] == []


def test_order_independent_and_inputs_unmodified(fixture_data):
    truth, pred = fixture_data
    originals = deepcopy(fixture_data)
    result = evaluate_detections(truth, pred, EvaluationConfig())
    assert fixture_data == originals
    for payload in fixture_data:
        payload["images"].reverse()
        for row in payload["images"]:
            row["boxes"].reverse()
    assert evaluate_detections(truth, pred, EvaluationConfig()) == result


@pytest.mark.parametrize("coords", [
    [10, 0, 0, 10], [0, 0, 10, 0], [-1, 0, 10, 10], [0, 0, 41, 10],
    [0, 0, 10, float("nan")], [0, 0, float("inf"), 10], [0, 0, 10],
    [False, 0, 10, 10], [0, 0, 10, "10"], [0, 0, 1e-200, 1e-200],
])
def test_reject_invalid_coordinates(fixture_data, coords):
    truth, pred = fixture_data
    for payload in (truth, pred):
        invalid = deepcopy(payload)
        invalid["images"][0]["boxes"][0]["xyxy"] = coords
        inputs = (invalid, pred) if payload is truth else (truth, invalid)
        with pytest.raises(ValueError, match="xyxy"):
            evaluate_detections(*inputs, EvaluationConfig())


@pytest.mark.parametrize("confidence", [None, -0.01, 1.01, float("nan"), True, "0.8"])
def test_invalid_confidence_rejected_even_below_threshold(fixture_data, confidence):
    truth, pred = fixture_data
    pred["images"][0]["boxes"][0]["confidence"] = confidence
    with pytest.raises(ValueError, match="confidence"):
        evaluate_detections(truth, pred, EvaluationConfig(confidence_threshold=1))


def test_invalid_roster_and_ground_truth_class(fixture_data):
    truth, pred = fixture_data
    pred["images"].append(pred["images"][0])
    with pytest.raises(ValueError, match="duplicate image id"):
        evaluate_detections(truth, pred, EvaluationConfig())
    pred["images"][-1] = {"image_id": "typo", "boxes": []}
    with pytest.raises(ValueError, match="absent from ground truth"):
        evaluate_detections(truth, pred, EvaluationConfig())
    pred["images"].pop()
    truth["images"][0]["boxes"][0]["class_name"] = "unknown"
    with pytest.raises(ValueError, match="undeclared class"):
        evaluate_detections(truth, pred, EvaluationConfig())


@pytest.mark.parametrize("iou,confidence", [(0, 0), (-1, 0), (1.1, 0), (0.5, True), (0.5, 2)])
def test_invalid_thresholds(iou, confidence):
    with pytest.raises(ValueError, match="threshold"):
        EvaluationConfig(iou, confidence)


def test_bad_json_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{x", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        load_json(path)
