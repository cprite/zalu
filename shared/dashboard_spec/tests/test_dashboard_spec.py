import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from shared.dashboard_spec.validation import (
    DashboardSpecError, SCHEMA, read_metric, validate_dashboard_spec,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SAFETY_CLASSES = ["person_without_vest", "person_with_vest"]


def load(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def safety():
    return load("workplace-safety")


@pytest.mark.parametrize("name,classes", [
    ("workplace-safety", SAFETY_CLASSES),
    ("parking-vehicles", ["car", "truck", "motorcycle"]),
])
def test_generated_example_round_trip(name, classes):
    Draft202012Validator.check_schema(SCHEMA)
    spec = json.loads(json.dumps(load(name)))
    validate_dashboard_spec(spec, class_names=classes)


def issues(spec):
    with pytest.raises(DashboardSpecError) as exc:
        validate_dashboard_spec(spec, class_names=SAFETY_CLASSES)
    return exc.value.errors


def test_missing_title(safety):
    del safety["title"]
    assert any("title" in issue["message"] for issue in issues(safety))


def test_unknown_widget_type(safety):
    safety["widgets"][0]["type"] = "employee_tracking"
    assert {"path": "/widgets/0/type", "message": "unsupported widget type 'employee_tracking'"} in issues(safety)


def test_widget_error_points_to_config(safety):
    safety["widgets"][0]["config"]["show_boxes"] = "yes"
    errors = issues(safety)
    assert all(issue["path"] == "/widgets/0/config/show_boxes" for issue in errors)


def test_reject_unregistered_sources_and_bindings(safety):
    safety["data_sources"][0]["type"] = "warehouse_api"
    assert any(issue["path"] == "/data_sources/0/type" for issue in issues(safety))
    safety["data_sources"][0]["type"] = "inference_session"
    safety["widgets"][0]["source_id"] = safety["metrics"][0]["source_id"] = "other_camera"
    assert {issue["path"] for issue in issues(safety)} >= {"/widgets/0/source_id", "/metrics/0/source_id"}


@pytest.mark.parametrize("change,expected", [
    ({"operation": "unique_visitors"}, "/metrics/0/operation"),
    ({"class_name": "forklift"}, "/metrics/0/class_name"),
    ({"class_name": None}, "/metrics/0/class_name"),
])
def test_reject_invalid_metrics(safety, change, expected):
    safety["metrics"][0].update(change)
    assert any(issue["path"] == expected for issue in issues(safety))


def test_metric_class_requirement_and_reference(safety):
    del safety["metrics"][0]["class_name"]
    assert any("class_name" in issue["message"] for issue in issues(safety))
    safety["metrics"][0]["class_name"] = SAFETY_CLASSES[0]
    safety["widgets"][1]["config"]["metric_id"] = "nonexistent"
    assert any(issue["path"] == "/widgets/1/config/metric_id" for issue in issues(safety))


def test_duplicate_ids_and_overflow(safety):
    safety["metrics"].append(safety["metrics"][0].copy())
    safety["widgets"][1]["id"] = safety["widgets"][0]["id"]
    safety["layout"]["columns"] = 6
    assert {item["path"] for item in issues(safety)} >= {"/metrics/2/id", "/widgets/1/id", "/widgets/0/span"}


def test_v1_minimal_dashboard_without_metrics():
    # Archived v1 payload. Keep it valid when the implementation grows.
    prior = json.loads("""{
        "version":"1", "title":"Camera", "description":"Current inference",
        "layout":{"columns":1}, "data_sources":[{"id":"session","type":"inference_session"}],
        "metrics":[], "widgets":[{"id":"camera","type":"live_preview","title":"Camera",
        "source_id":"session","span":1,"config":{"show_boxes":false}}]
    }""")
    validate_dashboard_spec(prior, class_names=[])
    prior["version"] = "2"
    assert any(issue["path"] == "/version" for issue in issues(prior))


def test_empty_frames_and_unavailable_metrics(safety):
    count, latency = safety["metrics"]
    unavailable = {"status": "unavailable", "value": None}
    assert read_metric(count, None, class_names=SAFETY_CLASSES) == unavailable
    assert read_metric(count, {}, class_names=SAFETY_CLASSES) == unavailable
    assert read_metric(count, {"detections": []}, class_names=[]) == unavailable
    assert read_metric(count, {"detections": []}, class_names=SAFETY_CLASSES) == {
        "status": "available", "value": 0,
    }
    assert read_metric(count, {"detections": [{"cls": SAFETY_CLASSES[0]}]}, class_names=SAFETY_CLASSES) == {
        "status": "available", "value": 1,
    }
    assert read_metric(latency, {"latency_ms": float("inf")}, class_names=SAFETY_CLASSES) == unavailable
    assert read_metric(latency, {"latency_ms": 0}, class_names=SAFETY_CLASSES) == {
        "status": "available", "value": 0,
    }
