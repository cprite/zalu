"""What a dashboard may show: DashboardSpec v1 limits + task/model vocabulary + Kolyan's data-source catalogue."""
from __future__ import annotations

from typing import List, Optional, Sequence

from pydantic import BaseModel

MAX_WINDOW_S = 600  # DashboardSpec v1 time_series window_seconds maximum

# Limits of DashboardSpec v1 itself (shared/dashboard_spec/README.md), independent of the backend.
V1_LIMITS = [
    "Counts are per camera frame (people/objects visible right now), not unique individuals: one person seen in "
    "many frames is counted in each frame.",
    f"Charts cover only the live session's last {MAX_WINDOW_S // 60} minutes; no daily/weekly totals or history.",
    "No occupancy or capacity percentages, dwell time, speed, identity or zone/region statistics.",
    "Event rows show only the latest frame's alerts; there is no stored incident log.",
]


class DashboardCapabilities(BaseModel):
    class_names: List[str]
    target_event: Optional[str]
    events_available: bool
    unavailable: List[str]

    def prompt_block(self) -> dict:
        return {
            "classes": self.class_names,
            "alert_class": self.target_event if self.events_available else None,
            "metric_operations": {
                "detection_count": "boxes of any class in the latest frame",
                "class_count": "boxes of one class in the latest frame",
                "inference_latency_ms": "model time per frame",
            },
            "widgets": {
                "live_preview": "camera image with boxes",
                "kpi": "one metric, current value",
                "time_series": f"one metric over the last window_seconds (max {MAX_WINDOW_S})",
                "detection_table": "latest frame rows: 'detections' or 'events' (events only if alert_class is set)",
                "class_distribution": "boxes per class in the latest frame",
            },
            "not_available": self.unavailable,
        }


def build_capabilities(task, model_classes: Optional[Sequence[str]] = None,
                       sources_catalog: Optional[dict] = None) -> DashboardCapabilities:
    """task: TaskSpec. model_classes: classes of the deployed model (trusted vocabulary; defaults to the task's).
    sources_catalog: response of GET /api/dashboard/sources (Kolyan), used to explain what is unavailable."""
    classes = list(model_classes) if model_classes else list(task.class_names)
    target = task.target_event if task.target_event in classes else None
    unavailable = list(V1_LIMITS)
    for s in (sources_catalog or {}).get("sources", []):
        if s.get("status") == "unavailable":
            unavailable.append(f"{s['name']}: {s.get('returns', 'not available')}")
        elif s.get("status") == "needs_processing":
            unavailable.append(f"{s['name']} (not live yet): {s.get('returns', '')}")
    if sources_catalog and any(s.get("name") == "counts.by_class" and s.get("status") == "supported"
                               for s in sources_catalog.get("sources", [])):
        unavailable.append("Whole-video summaries exist in the backend (video runs) but this dashboard format "
                           "(DashboardSpec v1) cannot display them yet.")
    return DashboardCapabilities(class_names=classes, target_event=target,
                                 events_available=target is not None, unavailable=unavailable)
