"""Validate generated DashboardSpec v1 and resolve allowlisted frame metrics."""

from collections.abc import Collection, Mapping
import json
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA = json.loads(Path(__file__).with_name("dashboard-spec.v1.schema.json").read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(SCHEMA)


class DashboardSpecError(ValueError):
    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{item['path'] or '/'}: {item['message']}" for item in errors))


def _issue(path, message):
    return {"path": path, "message": message}


def _schema_issue(error):
    if list(error.absolute_schema_path)[-3:] == ["widgets", "items", "oneOf"]:
        widget = error.instance
        if isinstance(widget, dict) and "type" in widget:
            for index, options in enumerate(SCHEMA["$defs"]["widget"]["oneOf"]):
                if widget["type"] == options["properties"]["type"]["const"]:
                    return [_schema_issue(child)[0] for child in error.context
                            if next(iter(child.schema_path), None) == index]
            path = "/" + "/".join(map(str, error.absolute_path))
            return [_issue(f"{path}/type", f"unsupported widget type {widget['type']!r}")]
        if isinstance(widget, dict):
            path = "/" + "/".join(map(str, error.absolute_path))
            return [_issue(f"{path}/type", "widget type is required")]
    parts = [str(p).replace("~", "~0").replace("/", "~1") for p in error.absolute_path]
    return [_issue("/" + "/".join(parts) if parts else "", error.message)]


def validate_dashboard_spec(spec: object, *, class_names: Collection[str]) -> None:
    """Raise DashboardSpecError with JSON Pointer paths; class_names comes from trusted task/model metadata."""
    errors = [issue for error in _VALIDATOR.iter_errors(spec) for issue in _schema_issue(error)]
    if errors:
        raise DashboardSpecError(errors)

    source_ids = {source["id"] for source in spec["data_sources"]}
    metrics = {}
    widget_ids = set()
    classes = set(class_names)
    for index, metric in enumerate(spec["metrics"]):
        path = f"/metrics/{index}"
        if metric["id"] in metrics:
            errors.append(_issue(f"{path}/id", "duplicate metric ID"))
        metrics[metric["id"]] = metric
        if metric["source_id"] not in source_ids:
            errors.append(_issue(f"{path}/source_id", "unknown data source"))
        if metric["operation"] == "class_count" and metric["class_name"] not in classes:
            errors.append(_issue(f"{path}/class_name", "class is absent from the selected model or task"))

    for index, widget in enumerate(spec["widgets"]):
        path = f"/widgets/{index}"
        if widget["id"] in widget_ids:
            errors.append(_issue(f"{path}/id", "duplicate widget ID"))
        widget_ids.add(widget["id"])
        if widget["source_id"] not in source_ids:
            errors.append(_issue(f"{path}/source_id", "unknown data source"))
        if widget["span"] > spec["layout"]["columns"]:
            errors.append(_issue(f"{path}/span", "span exceeds layout.columns"))
        if widget["type"] in ("kpi", "time_series"):
            metric = metrics.get(widget["config"]["metric_id"])
            if metric is None:
                errors.append(_issue(f"{path}/config/metric_id", "unknown metric"))
            elif metric["source_id"] != widget["source_id"]:
                errors.append(_issue(f"{path}/config/metric_id", "metric must use the widget's data source"))

    if errors:
        raise DashboardSpecError(errors)


def read_metric(
    metric: Mapping, frame: Mapping | None, *, class_names: Collection[str],
) -> dict:
    """Resolve a metric on one succeeded inference response, or return an unavailable reading."""
    unavailable = {"status": "unavailable", "value": None}
    if frame is None:
        return unavailable
    operation = metric["operation"]
    if operation == "inference_latency_ms":
        value = frame.get("latency_ms")
        if type(value) not in (float, int) or not 0 <= value <= 1e100:
            return unavailable
    elif operation in ("detection_count", "class_count"):
        detections = frame.get("detections")
        if not isinstance(detections, list) or any(
            not isinstance(box, Mapping) or not isinstance(box.get("cls"), str) for box in detections
        ):
            return unavailable
        if operation == "class_count":
            label = metric["class_name"]
            if label not in class_names:
                return unavailable
            value = sum(box["cls"] == label for box in detections)
        else:
            value = len(detections)
    else:
        raise ValueError(f"unsupported metric operation {operation!r}")
    return {"status": "available", "value": value}
