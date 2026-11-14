import json
import os
from pathlib import Path

import pytest

from backend.task_engine import TaskSpec
from backend.task_engine.dashboard import (DashboardDraft, PlannerUnavailable, build_capabilities, check_spec,
                                           fallback_spec, plan_dashboard, refine_dashboard, rule_refine)

ROOT = Path(__file__).resolve().parents[4]
EX = Path(__file__).resolve().parents[1] / "examples"
TASKS = Path(__file__).resolve().parents[2] / "examples"
CATALOG = json.loads((ROOT / "backend" / "examples" / "dashboard" / "sources.json").read_text())


@pytest.fixture
def vest():
    return TaskSpec.model_validate_json((TASKS / "vest.json").read_text())


@pytest.fixture
def parking():
    return TaskSpec.model_validate_json((TASKS / "parking.json").read_text())


def draft(**kw):
    base = dict(status="ok", title="T", description="D", columns=12,
                metrics=[{"id": "bad_now", "title": "No vest", "operation": "class_count", "class_name": "person_without_vest"}],
                widgets=[{"id": "cam", "type": "live_preview", "title": "Camera", "span": 8},
                         {"id": "k", "type": "kpi", "title": "No vest", "span": 4, "metric_id": "bad_now"}],
                limitations=[], clarification_questions=[], rationale="r")
    base.update(kw)
    return DashboardDraft.model_validate(base)


class FakeLLM:
    name = "fake"

    def __init__(self, *outs):
        self.outs, self.calls = list(outs), []

    def __call__(self, system, messages):
        self.calls.append(messages)
        out = self.outs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_valid_draft_becomes_valid_spec(vest):
    r = plan_dashboard("alert on no vest", vest, llm=FakeLLM(draft()))
    assert r.ai_generated and r.status == "ok" and r.generator == "fake"
    assert r.spec["version"] == "1" and r.spec["data_sources"] == [{"id": "live", "type": "inference_session"}]
    assert check_spec(r.spec, build_capabilities(vest)) == []


def test_long_window_is_clamped_and_reported(vest):
    d = draft(widgets=[{"id": "ts", "type": "time_series", "title": "Today", "span": 12, "metric_id": "bad_now",
                        "window_seconds": 86400}])
    r = plan_dashboard("violations today", vest, llm=FakeLLM(d))
    assert r.spec["widgets"][0]["config"]["window_seconds"] == 600
    assert r.status == "partial" and any("10 minutes" in l for l in r.limitations)


def test_unknown_class_triggers_repair_then_fallback(vest):
    bad = draft(metrics=[{"id": "fork", "title": "Forklifts", "operation": "class_count", "class_name": "forklift"}],
                widgets=[{"id": "k", "type": "kpi", "title": "F", "span": 4, "metric_id": "fork"}])
    llm = FakeLLM(bad, draft())
    r = plan_dashboard("forklifts", vest, llm=llm)
    assert r.ai_generated and "Invalid" in llm.calls[1][-1]["content"]

    r2 = plan_dashboard("forklifts", vest, llm=FakeLLM(bad, bad))
    assert not r2.ai_generated and r2.generator == "fallback:rules" and "invalid" in r2.fallback_reason


def test_events_table_needs_alert_class(parking):
    d = draft(metrics=[], widgets=[{"id": "t", "type": "detection_table", "title": "Events", "span": 6, "rows": "events"}])
    r = plan_dashboard("cars", parking, llm=FakeLLM(d, d))
    assert not r.ai_generated  # parking has no alert class -> events rejected twice -> fallback
    assert all(w["config"].get("rows") != "events" for w in r.spec["widgets"])


def test_api_unavailable_gives_labelled_fallback(vest):
    r = plan_dashboard("vests", vest, llm=FakeLLM(PlannerUnavailable("APIConnectionError: down")))
    assert r.ai_generated is False and r.generator == "fallback:rules" and "down" in r.fallback_reason
    assert any("AI planner unavailable" in l for l in r.limitations)
    assert check_spec(r.spec, build_capabilities(vest)) == []


def test_fallback_layouts_differ_by_task(vest, parking):
    s_spec, s_kind = fallback_spec(vest, build_capabilities(vest))
    p_spec, p_kind = fallback_spec(parking, build_capabilities(parking))
    assert (s_kind, p_kind) == ("safety", "counting")
    assert [w["type"] for w in s_spec["widgets"]][:2] == ["kpi", "live_preview"]  # alert first
    assert [w["type"] for w in p_spec["widgets"]][:2] == ["live_preview", "kpi"]  # camera first
    assert any(w["config"].get("rows") == "events" for w in s_spec["widgets"])
    assert not any(w["config"].get("rows") == "events" for w in p_spec["widgets"])
    for spec, task in ((s_spec, vest), (p_spec, parking)):
        assert check_spec(spec, build_capabilities(task)) == []


def test_not_ok_task_returns_fallback_with_issues(vest):
    t = vest.model_copy(update={"status": "needs_clarification", "issues": ["what to detect?"]})
    r = plan_dashboard("watch the shop", t, llm=FakeLLM())
    assert not r.ai_generated and "what to detect?" in r.limitations


def test_capabilities_use_catalogue_and_model_classes(vest):
    caps = build_capabilities(vest, model_classes=["person_with_vest"], sources_catalog=CATALOG)
    assert caps.class_names == ["person_with_vest"] and caps.events_available is False  # target not in model
    joined = " ".join(caps.unavailable)
    assert "counts.unique_objects" in joined and "occupancy.percent" in joined and "cannot display them" in joined


@pytest.mark.parametrize("instruction,check", [
    ("Add a chart of detections over time.", lambda old, new: len(new["widgets"]) == len(old["widgets"]) + 1
     and new["widgets"][-1]["type"] == "time_series"),
    ("Make the video larger.", lambda old, new: next(w for w in new["widgets"] if w["type"] == "live_preview")["span"]
     > next(w for w in old["widgets"] if w["type"] == "live_preview")["span"]),
    ("Remove the event table.", lambda old, new: not any(w["type"] == "detection_table" and w["config"]["rows"] == "events"
                                                         for w in new["widgets"])),
])
def test_rule_refine_examples(vest, instruction, check):
    caps = build_capabilities(vest)
    spec, _ = fallback_spec(vest, caps)
    new, applied = rule_refine(spec, instruction, caps)
    assert applied and check(spec, new) and check_spec(new, caps) == []


def test_refine_falls_back_to_rules_when_api_down(vest):
    caps = build_capabilities(vest)
    spec, _ = fallback_spec(vest, caps)
    r = refine_dashboard(spec, "Remove the event table.", vest, llm=FakeLLM(PlannerUnavailable("timeout")))
    assert r.status == "ok" and not r.ai_generated and "removed" in r.rationale
    r2 = refine_dashboard(spec, "Rearrange it so it feels calmer", vest, llm=FakeLLM(PlannerUnavailable("timeout")))
    assert r2.status == "partial" and r2.spec == spec


def test_refine_rejects_invalid_current_spec(vest):
    with pytest.raises(ValueError):
        refine_dashboard({"version": "1"}, "bigger video", vest, llm=FakeLLM())


@pytest.mark.parametrize("name,task", [("safety_vest", "vest"), ("parking_vehicles", "parking"),
                                       ("vest_hourly_partial", "vest")])
def test_saved_live_examples_are_valid(name, task, request):
    t = request.getfixturevalue(task)
    plan = json.loads((EX / f"{name}.plan.json").read_text())
    assert plan["ai_generated"] and check_spec(plan["spec"], build_capabilities(t)) == []


def test_codex_fixtures_pass_our_checks(vest):
    spec = json.loads((ROOT / "shared" / "dashboard_spec" / "examples" / "workplace-safety.json").read_text())
    assert check_spec(spec, build_capabilities(vest)) == []


@pytest.mark.skipif(not os.environ.get("VF_LIVE_TESTS"), reason="set VF_LIVE_TESTS=1 to call Claude")
def test_live_personalisation(vest, parking):
    s = plan_dashboard("Detect workers without safety vests and alert me.", vest)
    p = plan_dashboard("Detect vehicles in my parking lot and show their counts over time.", parking)
    assert s.ai_generated and p.ai_generated
    assert [w["type"] for w in s.spec["widgets"]] != [w["type"] for w in p.spec["widgets"]]
