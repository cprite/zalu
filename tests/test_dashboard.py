"""Deterministic tests for backend/dashboard.py (no GPU, no network)."""
import json
from pathlib import Path

from backend import dashboard, schemas


def _run(seq):
    """seq: list of lists of class names per sampled frame (2 fps)."""
    frames = [schemas.FrameDetections(frame_index=i * 12, t_s=i * 0.5,
              detections=[schemas.Detection(cls=c, score=0.9, box=[0, 0, 10, 10]) for c in cs]) for i, cs in enumerate(seq)]
    return schemas.VideoRun(model_id="model_x", source_name="t.mp4", fps_sampled=2, conf=0.25, frames=frames)


def _task():
    ex = json.loads((Path(__file__).parents[1] / "backend/task_engine/examples/vest.json").read_text())
    for k in ("prompt", "task_id", "compiler"):
        ex.pop(k, None)
    return schemas.TaskSpec(**ex, prompt="vest", compiler="test")


def test_summary_counts_are_detections_not_objects():
    s = dashboard.summary(_run([["a", "a"], [], ["a", "b"]]))
    assert s["per_class"]["a"] == {"detections": 3, "frames_with_class": 2, "share_of_frames": 0.667,
                                   "max_in_one_frame": 2, "mean_per_frame": 1.0}
    assert "not unique objects" in s["caveat"]


def test_timeseries_max_per_bucket():
    t = dashboard.timeseries(_run([["a"], ["a", "a"], [], ["b"]]), bucket_s=1.0)
    assert t["points"] == [{"t_s": 0.0, "a": 2, "b": 0}, {"t_s": 1.0, "a": 0, "b": 1}]


def test_events_need_rules_and_consecutive_frames():
    task = _task()
    rule = task.event_rules[0]
    trig, need = rule.trigger_class, rule.min_consecutive_frames
    seq = [[trig]] * (need - 1) + [[]] + [[trig]] * need + [[]]
    e = dashboard.events(_run(seq), task)
    assert e["supported"] and len(e["events"]) == 1
    assert e["events"][0]["frames"] == need and e["events"][0]["start_s"] == need * 0.5
    assert dashboard.events(_run(seq), None)["supported"] is False
