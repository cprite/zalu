"""Dashboard data: aggregations over VideoRuns (Kolyan). Pure functions, no GPU, no invented numbers.

Rules (from the brief, enforced here):
- detections are per frame; they are NOT unique objects (no tracking yet) -> counts are reported as
  'detections', 'frames_with_class' and 'max_in_one_frame', never as 'people' or 'unique objects';
- an event exists only if the TaskSpec defines an EventRule (trigger_class + min_consecutive_frames);
- occupancy % is unavailable (needs a known capacity or defined regions);
- time series use VIDEO time (t_s), not wall-clock time.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from backend import schemas

SOURCES = [
    # name, status, endpoint, what it returns / why not
    {"name": "model.info", "status": "supported", "endpoint": "GET /api/models/{model_id}",
     "returns": "ModelArtifact: classes, base model, val metrics, baseline metrics, train_seconds, onnx path, export check"},
    {"name": "model.status", "status": "supported", "endpoint": "GET /api/dashboard/models/{model_id}/status",
     "returns": "deployable (weights/onnx present), runs on this model, last run time, latest evaluation report id"},
    {"name": "detections.frame", "status": "supported", "endpoint": "POST /api/inference",
     "returns": "one image -> detections (class, score, box xyxy px), latency_ms, events for target_event"},
    {"name": "detections.video", "status": "supported", "endpoint": "POST /api/runs  then  GET /api/runs/{run_id}",
     "returns": "VideoRun: per sampled frame {frame_index, t_s, detections[]}"},
    {"name": "counts.by_class", "status": "supported", "endpoint": "GET /api/runs/{run_id}/summary",
     "returns": "per class: detections, frames_with_class, max_in_one_frame, mean_per_frame. NOT unique objects"},
    {"name": "timeseries.class_counts", "status": "supported", "endpoint": "GET /api/runs/{run_id}/timeseries?bucket_s=1",
     "returns": "per bucket of video time: max detections per class in one frame"},
    {"name": "events", "status": "supported_if_task_has_event_rules", "endpoint": "GET /api/runs/{run_id}/events",
     "returns": "intervals where the TaskSpec trigger_class is present for >= min_consecutive_frames sampled frames"},
    {"name": "evaluation.report", "status": "supported", "endpoint": "GET /api/reports/{report_id}",
     "returns": "Damir's EvaluationReport: per model P/R/F1 with 95% CI, per class, readiness verdict"},
    {"name": "counts.unique_objects", "status": "unavailable", "endpoint": None,
     "returns": "needs multi-object tracking across frames (not implemented)"},
    {"name": "occupancy.percent", "status": "unavailable", "endpoint": None,
     "returns": "needs a known capacity or defined regions (zones) per camera"},
    {"name": "history.wall_clock", "status": "needs_processing", "endpoint": "GET /api/runs (list)",
     "returns": "runs are retained with created_at, but there is no continuous live-camera logging yet"},
]


def summary(run: schemas.VideoRun) -> dict:
    n = len(run.frames)
    det, frames_with, mx = defaultdict(int), defaultdict(int), defaultdict(int)
    for f in run.frames:
        per = defaultdict(int)
        for d in f.detections:
            per[d.cls] += 1
        for c, k in per.items():
            det[c] += k
            frames_with[c] += 1
            mx[c] = max(mx[c], k)
    classes = sorted(det)
    return {
        "run_id": run.id, "model_id": run.model_id, "source_name": run.source_name,
        "sampled_frames": n, "fps_sampled": run.fps_sampled, "duration_s": run.duration_s, "conf": run.conf,
        "per_class": {c: {"detections": det[c], "frames_with_class": frames_with[c],
                          "share_of_frames": round(frames_with[c] / n, 3) if n else 0,
                          "max_in_one_frame": mx[c], "mean_per_frame": round(det[c] / n, 3) if n else 0}
                      for c in classes},
        "frames_with_any_detection": sum(1 for f in run.frames if f.detections),
        "caveat": "Counts are per-frame detections, not unique objects (no tracking).",
    }


def timeseries(run: schemas.VideoRun, bucket_s: float = 1.0) -> dict:
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in run.frames:
        b = int(f.t_s // bucket_s)
        per = defaultdict(int)
        for d in f.detections:
            per[d.cls] += 1
        for c, k in per.items():
            buckets[b][c] = max(buckets[b][c], k)
        buckets[b]  # keep empty buckets visible
    classes = sorted({c for v in buckets.values() for c in v})
    pts = [{"t_s": round(b * bucket_s, 3), **{c: buckets[b].get(c, 0) for c in classes}} for b in sorted(buckets)]
    return {"run_id": run.id, "bucket_s": bucket_s, "metric": "max detections per class in one sampled frame",
            "time_base": "video seconds", "classes": classes, "points": pts}


def events(run: schemas.VideoRun, task: Optional[schemas.TaskSpec]) -> dict:
    rules = list(task.event_rules) if task else []
    if not rules:
        return {"run_id": run.id, "supported": False, "events": [],
                "reason": "the task has no event rules (TaskSpec.event_rules), so no events are defined"}
    out = []
    for r in rules:
        streak, start = 0, None
        for idx, f in enumerate(run.frames + [None]):
            hit = f is not None and any(d.cls == r.trigger_class for d in f.detections)
            if hit:
                streak += 1
                start = f.t_s if streak == 1 else start
                last = f
            else:
                if streak >= r.min_consecutive_frames:
                    peak = max((d.score for g in run.frames if start <= g.t_s <= last.t_s for d in g.detections
                                if d.cls == r.trigger_class), default=None)
                    out.append({"rule": r.description, "trigger_class": r.trigger_class, "start_s": start,
                                "end_s": last.t_s, "frames": streak, "peak_score": peak,
                                "frame_index_start": next(g.frame_index for g in run.frames if g.t_s == start)})
                streak = 0
    return {"run_id": run.id, "supported": True, "events": out,
            "rules": [{"trigger_class": r.trigger_class, "min_consecutive_frames": r.min_consecutive_frames,
                       "description": r.description} for r in rules],
            "note": "min_consecutive_frames is applied to SAMPLED frames (fps_sampled), not every video frame."}
