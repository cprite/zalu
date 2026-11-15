"""Deterministic YOLO adapter and scoring contracts."""

from types import SimpleNamespace

import pytest

from backend import schemas
from backend.evaluation_engine import evaluate_model
from backend.evaluation_engine.yolo import from_inference_result, from_ultralytics_result, parse_yolo_labels


class TensorFixture(list):
    def tolist(self):
        return list(self)


def raw_result(coords, scores, ids):
    return SimpleNamespace(
        orig_shape=(32, 64), names={7: "worker", 2: "vest"},
        boxes=SimpleNamespace(xyxy=TensorFixture(coords), conf=TensorFixture(scores), cls=TensorFixture(ids)),
    )


def test_normalized_labels_and_raw_yolo_share_pixel_space_and_class_mapping():
    size = (64, 32)
    labels = parse_yolo_labels("frame", "0 0.25 0.5 0.5 1\n1 0.75 0.5 0.25 0.5", size, ["person", "vest"])
    assert [b.box for b in labels] == [(0, 0, 32, 32), (40, 8, 56, 24)]
    result = raw_result([[40, 8, 56, 24], [0, 0, 32, 32]], [0.75, 0.5], [2.0, 7.0])
    preds = from_ultralytics_result("frame", result, size=size, class_map={"worker": "person"})
    report = evaluate_model(labels, preds, ["frame"], ["person", "vest"], iou_threshold=1, conf_threshold=0.5)
    assert report["overall"]["tp"] == 2
    assert report["overall"]["f1"] == 1
    assert not report["failures"]


def test_empty_inference_and_yolo_edge_rounding():
    size = (64, 32)
    labels = parse_yolo_labels("frame", "0 0.166666 0.5 0.333333 1", size, ["vest"])
    assert labels[0].box[0] == 0
    assert parse_yolo_labels("frame", "", size, ["vest"]) == []
    preds = from_ultralytics_result("frame", raw_result([], [], []), size=size)
    scored = evaluate_model(labels, preds, ["frame"], ["vest"])
    assert scored["overall"]["fn"] == 1


@pytest.mark.parametrize("row", [
    "-1 0.25 0.5 0.5 1", "1 0.25 0.5 0.5 1", "0 0.25 nan 0.5 1",
    "0 0.25 0.5 0 1", "0 0.25 0.5 0.6 1", "0 0.25 0.5 0.5 1 0.99",
])
def test_invalid_label_includes_file_and_line(row):
    with pytest.raises(ValueError, match="frame.txt:1"):
        parse_yolo_labels("frame", row, (64, 32), ["person"], label_path="frame.txt")


@pytest.mark.parametrize("class_id", [-1, 7.5, 9, float("nan")])
def test_invalid_raw_class_ids(class_id):
    with pytest.raises(ValueError, match="model class ID"):
        from_ultralytics_result(
            "frame", raw_result([[0, 0, 32, 32]], [0.5], [class_id]), size=(64, 32),
        )


def test_raw_result_requires_aligned_arrays():
    with pytest.raises(ValueError, match="array lengths"):
        from_ultralytics_result("frame", raw_result([], [0.5], []), size=(64, 32))


def test_worker_shared_model_contract():
    result = schemas.InferenceResult(
        model_id="model_fixture", width=64, height=32, latency_ms=0,
        detections=[schemas.Detection(cls="worker", box=[0, 0, 32, 32], score=0.5)],
    )
    boxes = from_inference_result("frame", result, size=(64, 32), class_map={"worker": "person"})
    assert boxes[0].cls == "person" and boxes[0].score == 0.5
    with pytest.raises(ValueError, match="different model ID"):
        from_inference_result("frame", result, size=(64, 32), model_id="different")


@pytest.mark.parametrize("coords", [[0, 0, 65, 32], [32, 0, 0, 32], [0, 0, 32, 0], [0, 0, float("inf"), 32]])
def test_invalid_worker_box(coords):
    packet = {"width": 64, "height": 32, "detections": [{"cls": "person", "box": coords, "score": 0.5}]}
    with pytest.raises(ValueError, match="xyxy"):
        from_inference_result("frame", packet, size=(64, 32))
