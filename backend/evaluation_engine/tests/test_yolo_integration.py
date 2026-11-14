"""Exercise label files, YOLO-shaped predictions, metrics and API payloads end to end."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from backend.evaluation_engine import EvaluationConfig, load_json, select_on_validation
from backend.evaluation_engine.examples.yolo.generate import OUTPUT, build_example
from backend.evaluation_engine.yolo import (
    YoloLabelImage, build_test_per_model, evaluate_yolo, from_inference_result,
    from_ultralytics_result, load_yolo_ground_truth,
)


DIRECTORY = Path(__file__).resolve().parents[1]


class FakeTensor(list):
    def tolist(self):
        return list(self)


def result(coords, scores, classes):
    return SimpleNamespace(
        names={3: "vest", 9: "human"}, orig_shape=(32, 64),
        boxes=SimpleNamespace(
            xyxy=FakeTensor(coords), conf=FakeTensor(scores), cls=FakeTensor(classes),
        ),
    )


@pytest.fixture
def labelled(tmp_path):
    path = tmp_path / "image.txt"
    path.write_text("1 0.25 0.5 0.5 1\n0 0.75 0.5 0.25 0.5\n", encoding="utf-8")
    return YoloLabelImage("image", path, 64, 32)


def ground_truth(labelled, split="validation"):
    return load_yolo_ground_truth(
        [labelled], dataset_id="ds_fixture", split=split,
        source="synthetic_fixture", class_names=["vest", "person"],
    )


def test_ultralytics_matches_converted_labels_and_reordered_class_ids(labelled):
    truth = ground_truth(labelled)
    assert truth["images"][0]["boxes"] == [
        {"class_name": "person", "xyxy": [0, 0, 32, 32]},
        {"class_name": "vest", "xyxy": [40, 8, 56, 24]},
    ]
    raw = result([[40, 8, 56, 24], [0, 0, 32, 32]], [0.75, 0.5], [3.0, 9.0])
    row = from_ultralytics_result("image", raw, class_map={"human": "person"})
    assert row["width"] == 64 and row["height"] == 32
    scored = evaluate_yolo(truth, [row], EvaluationConfig(1, 0.5), model_id="model_fixture")
    assert scored["summary"]["tp"] == 2
    assert scored["summary"]["f1"] == 1.0
    assert select_on_validation([scored]) == scored
    final = evaluate_yolo({**truth, "split": "test"}, [row], EvaluationConfig(), model_id="model_fixture")
    with pytest.raises(ValueError, match="require validation"):
        select_on_validation([final])


def test_empty_detections_and_unmatched_model_class(labelled):
    truth = ground_truth(labelled)
    empty = from_ultralytics_result("image", result([], [], []))
    scored = evaluate_yolo(truth, [empty], EvaluationConfig(), model_id="empty")
    assert scored["summary"]["fn"] == 2
    assert scored["summary"]["images_without_prediction_rows"] == 0
    unknown = from_ultralytics_result("image", result([[0, 0, 32, 32]], [0.9], [9]))
    scored = evaluate_yolo(truth, [unknown], EvaluationConfig(), model_id="unknown_class")
    assert scored["per_class"]["human"]["fp"] == 1
    assert scored["error_counts"]["class_mismatch"] == 1


@pytest.mark.parametrize("class_id", [-1, 3.8, 7])
def test_prediction_class_id_validation(class_id):
    with pytest.raises(ValueError, match="class ID"):
        from_ultralytics_result("image", result([[0, 0, 1, 1]], [0.9], [class_id]))


def test_ultralytics_requires_boxes_and_equal_lengths():
    with pytest.raises(ValueError, match="array lengths"):
        from_ultralytics_result("image", result([], [0.5], []))
    with pytest.raises(ValueError, match="detection result"):
        from_ultralytics_result("image", SimpleNamespace(boxes=None))


@pytest.mark.parametrize("line", [
    "2 0.25 0.5 0.5 1", "0 0.25 0.5 0.5", "0 0.25 nan 0.5 1",
    "0 0.25 0.5 0 1", "0 0.25 0.5 0.6 1", "1.5 0.25 0.5 0.5 1",
])
def test_bad_label_has_file_and_line_context(labelled, line):
    labelled.label_path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="image.txt:1"):
        ground_truth(labelled)


def test_missing_label_is_distinct_from_background(labelled):
    labelled.label_path.write_text("", encoding="utf-8")
    assert ground_truth(labelled)["images"][0]["boxes"] == []
    labelled.label_path.unlink()
    with pytest.raises(ValueError, match="cannot read YOLO labels"):
        ground_truth(labelled)


def test_absorbs_six_decimal_edge_rounding(labelled):
    labelled.label_path.write_text("0 0.166666 0.5 0.333333 1", encoding="utf-8")
    assert ground_truth(labelled)["images"][0]["boxes"][0]["xyxy"][0] == 0


@pytest.mark.parametrize("case,match", [
    ("missing", "missing inference"), ("duplicate", "duplicate prediction"),
    ("extra", "absent from YOLO ground truth"), ("dimensions", "dimensions differ"),
])
def test_evaluation_requires_complete_dimension_checked_inference(labelled, case, match):
    truth = ground_truth(labelled)
    row = from_inference_result("image", {"width": 64, "height": 32, "detections": []})
    rows = {
        "missing": [], "duplicate": [row, row], "extra": [{**row, "image_id": "other"}],
        "dimensions": [{**row, "width": 32}],
    }[case]
    with pytest.raises(ValueError, match=match):
        evaluate_yolo(truth, rows, EvaluationConfig(), model_id="model_fixture")


def test_api_example_round_trips_and_reproduces_exact_metrics():
    payload = build_example()
    assert payload == load_json(OUTPUT)
    assert payload["mock"] is True
    assert payload["request"]["model_ids"] == list(payload["per_model"])
    baseline, trained = payload["per_model"].values()
    schema = Draft202012Validator(load_json(DIRECTORY / "evaluation_report.schema.json"))
    for entry in (baseline, trained):
        schema.validate(entry)
        assert entry["dataset"]["source"] == "synthetic_fixture"
        assert entry["readiness"]["status"] == "ineligible"
    assert baseline["summary"]["tp"] == 1
    assert baseline["summary"]["fp"] == 2
    assert baseline["summary"]["fn"] == 1
    assert baseline["summary"]["f1"] == 2 / 5
    assert baseline["failures"][0]["errors"][0]["type"] == "class_mismatch"
    assert baseline["failures"][1]["ground_truth"] == []
    assert trained["summary"]["f1"] == 1


def test_final_test_envelope_rejects_validation_synthetic_and_incompatible_reports():
    example = build_example()
    request = example["request"]
    dataset_id = request["test_dataset_id"]
    model_ids = request["model_ids"]
    reports = list(example["per_model"].values())
    with pytest.raises(ValueError, match="independently labelled real test dataset"):
        build_test_per_model(reports, dataset_id=dataset_id, model_ids=model_ids)

    # Mock trusted upstream provenance solely to exercise the real API boundary.
    eligible = deepcopy(reports)
    for entry in eligible:
        entry["dataset"]["source"] = "independently_labelled_real"
    assert build_test_per_model(eligible, dataset_id=dataset_id, model_ids=model_ids) == dict(
        zip(model_ids, eligible)
    )
    with pytest.raises(ValueError, match="differ from requested"):
        build_test_per_model(eligible[:1], dataset_id=dataset_id, model_ids=model_ids)
    invalid = deepcopy(eligible)
    invalid[1]["dataset"]["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="share the ground truth"):
        build_test_per_model(invalid, dataset_id=dataset_id, model_ids=model_ids)
    eligible[1]["dataset"]["split"] = "validation"
    with pytest.raises(ValueError, match="independently labelled real test dataset"):
        build_test_per_model(eligible, dataset_id=dataset_id, model_ids=model_ids)
