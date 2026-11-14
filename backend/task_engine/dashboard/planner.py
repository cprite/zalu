"""Dashboard Planner: user prompt + TaskSpec + capabilities -> validated DashboardSpec v1.

    plan_dashboard(prompt, task, ...)              -> PlannerResult
    refine_dashboard(spec, instruction, task, ...) -> PlannerResult   (no retraining; same data only)

Claude fills a flat DashboardDraft; this module converts it to DashboardSpec v1, enforces limits
(window <= 600 s, known classes, events only with an alert class), validates with Codex's
shared validator, allows one repair round, and otherwise falls back to a deterministic layout.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, List, Optional, Tuple

from shared.dashboard_spec.validation import DashboardSpecError, validate_dashboard_spec

from .capabilities import MAX_WINDOW_S, DashboardCapabilities, build_capabilities
from .schema import DashboardDraft, MetricDraft, PlannerResult, WidgetDraft

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("VF_DASHBOARD_MODEL", "claude-opus-5-5")
SOURCE_ID = "live"
ID_RE = re.compile(r"[^a-z0-9_-]+")

# (system, messages) -> DashboardDraft
DraftBackend = Callable[[str, List[dict]], DashboardDraft]


class PlannerUnavailable(RuntimeError):
    """The LLM could not be used (network, auth, timeout, refusal, truncation)."""


SYSTEM_PROMPT = """You design monitoring dashboards for ZALU, a tool that turns a camera into a trained object detector.
Output a DashboardDraft. Use ONLY the classes, metric operations and widgets listed in `capabilities`; never invent
data, classes or statistics.

- Put what the user cares about most first (widgets are rendered in order on a `columns`-wide grid; span = width).
- Safety/alert goals: lead with the alert-class KPI and events, then the live view and a trend.
  Counting/monitoring goals: lead with the live view and counts, then trends and the per-class breakdown.
- Title widgets and metrics in the user's own words ("Workers without a vest", "Cars in the lot").
- If the user asks for something listed in `not_available` (unique people, daily totals, occupancy %, history,
  zones...), do not fake it: show the closest honest proxy only if it is genuinely useful (e.g. per-frame count over
  the last 10 minutes) and explain the gap in `limitations`; status "partial". If nothing requested can be shown,
  status "unsupported" with a minimal live view.
- 3-7 widgets. Keep text short."""

REFINE_SYSTEM = SYSTEM_PROMPT + """

You are editing an existing dashboard. Apply the user's instruction to `current_dashboard` and return the complete
updated draft. Keep everything the instruction does not mention unchanged (ids, titles, order). The vision model
cannot be retrained here: if the instruction needs a class or statistic that is not available, keep the dashboard,
status "partial", and say what retraining or backend capability it would need in `limitations`."""


# ---------------------------------------------------------------- LLM backend
def claude_draft_backend(model: str = DEFAULT_MODEL, client=None, timeout: float = 45.0,
                         max_retries: int = 2) -> DraftBackend:
    """Anthropic SDK backend (uses ANTHROPIC_API_KEY on the server; never sent to the browser)."""
    import anthropic

    client = client or anthropic.Anthropic(timeout=timeout, max_retries=max_retries)

    def call(system: str, messages: List[dict]) -> DashboardDraft:
        try:
            resp = client.beta.messages.parse(
                model=model, max_tokens=8000, system=system, messages=messages,
                output_format=DashboardDraft, output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.APIError as e:  # connection, timeout, 4xx/5xx after SDK retries
            raise PlannerUnavailable(f"{type(e).__name__}: {e}") from e
        if resp.stop_reason == "refusal":
            raise PlannerUnavailable("Claude declined the request")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise PlannerUnavailable(f"no usable output (stop_reason={resp.stop_reason})")
        return resp.parsed_output

    call.name = f"claude:{model}"  # type: ignore[attr-defined]
    return call


# ---------------------------------------------------------------- draft -> DashboardSpec v1
def _slug(s: str, fallback: str) -> str:
    s = ID_RE.sub("_", (s or "").lower()).strip("_-")
    if not s or not s[0].isalpha():
        s = f"{fallback}_{s}".strip("_")
    return s[:48]


def to_spec(draft: DashboardDraft, caps: DashboardCapabilities) -> Tuple[dict, List[str]]:
    """Convert a draft into DashboardSpec v1, clamping to legal ranges. Returns (spec, auto_limitations)."""
    notes: List[str] = []
    cols = min(12, max(1, draft.columns or 12))
    metrics, mids = [], {}
    for i, m in enumerate(draft.metrics):
        mid = _slug(m.id, f"m{i}")
        while mid in mids.values():
            mid = f"{mid[:44]}_{i}"
        mids[m.id] = mid
        md = {"id": mid, "title": m.title[:100] or mid, "source_id": SOURCE_ID, "operation": m.operation}
        if m.operation == "class_count":
            md["class_name"] = m.class_name or ""
        metrics.append(md)

    widgets, wids = [], set()
    for i, w in enumerate(draft.widgets):
        wid = _slug(w.id, f"w{i}")
        while wid in wids:
            wid = f"{wid[:44]}_{i}"
        wids.add(wid)
        base = {"id": wid, "type": w.type, "title": w.title[:100] or wid, "source_id": SOURCE_ID,
                "span": min(cols, max(1, w.span or cols))}
        if w.type == "live_preview":
            base["config"] = {"show_boxes": True if w.show_boxes is None else bool(w.show_boxes)}
        elif w.type in ("kpi", "time_series"):
            base["config"] = {"metric_id": mids.get(w.metric_id or "", w.metric_id or "")}
            if w.type == "time_series":
                win = w.window_seconds or 300
                if win > MAX_WINDOW_S:
                    notes.append(f"'{w.title}' asked for {win} s of history; live dashboards keep at most "
                                 f"{MAX_WINDOW_S // 60} minutes, so it shows the last {MAX_WINDOW_S // 60} minutes.")
                base["config"]["window_seconds"] = min(MAX_WINDOW_S, max(5, win))
        elif w.type == "detection_table":
            base["config"] = {"rows": w.rows or "detections", "max_rows": min(100, max(1, w.max_rows or 20))}
        else:
            base["config"] = {}
        widgets.append(base)

    spec = {"version": "1", "title": (draft.title or "Camera dashboard")[:100],
            "description": (draft.description or draft.title or "Camera dashboard")[:500],
            "layout": {"columns": cols}, "data_sources": [{"id": SOURCE_ID, "type": "inference_session"}],
            "metrics": metrics, "widgets": widgets}
    return spec, notes


def check_spec(spec: dict, caps: DashboardCapabilities) -> List[str]:
    """Codex's validator plus capability checks. Returns human-readable errors (empty = valid)."""
    errors: List[str] = []
    try:
        validate_dashboard_spec(spec, class_names=caps.class_names)
    except DashboardSpecError as e:
        errors += [f"{x['path'] or '/'}: {x['message']}" for x in e.errors]
    for i, w in enumerate(spec.get("widgets", [])):
        if w.get("type") == "detection_table" and w.get("config", {}).get("rows") == "events" and not caps.events_available:
            errors.append(f"/widgets/{i}/config/rows: events are unavailable (the task has no alert class); use 'detections'")
    return errors


# ---------------------------------------------------------------- deterministic fallback
def fallback_spec(task, caps: DashboardCapabilities) -> Tuple[dict, str]:
    """Rule-based layout from the TaskSpec alone. Safety tasks lead with alerts; others with the live view."""
    target = caps.target_event
    others = [c for c in caps.class_names if c != target]
    m = [MetricDraft(id="all_detections", title="Objects in view", operation="detection_count")]
    w: List[WidgetDraft] = []
    if target:
        pretty = target.replace("_", " ")
        m.append(MetricDraft(id="alerts_now", title=f"{pretty.capitalize()} (now)", operation="class_count", class_name=target))
        if others:
            m.append(MetricDraft(id="ok_now", title=f"{others[0].replace('_', ' ').capitalize()} (now)",
                                 operation="class_count", class_name=others[0]))
        w += [WidgetDraft(id="alerts_kpi", type="kpi", title=f"{pretty.capitalize()} now", span=4, metric_id="alerts_now"),
              WidgetDraft(id="camera", type="live_preview", title="Live camera", span=8, show_boxes=True),
              WidgetDraft(id="alerts_trend", type="time_series", title=f"{pretty.capitalize()}, last 5 min", span=8,
                          metric_id="alerts_now", window_seconds=300),
              WidgetDraft(id="alerts_table", type="detection_table", title="Current alerts", span=4, rows="events", max_rows=10)]
        if others:
            w.append(WidgetDraft(id="ok_kpi", type="kpi", title=m[-1].title, span=4, metric_id="ok_now"))
        w.append(WidgetDraft(id="mix", type="class_distribution", title="What the camera sees", span=8))
        title, kind = "Safety monitor", "safety"
    else:
        for c in caps.class_names[:3]:
            m.append(MetricDraft(id=_slug(f"{c}_now", "c"), title=f"{c.replace('_', ' ').capitalize()} (now)",
                                 operation="class_count", class_name=c))
        w += [WidgetDraft(id="camera", type="live_preview", title="Live camera", span=7, show_boxes=True),
              WidgetDraft(id="total_kpi", type="kpi", title="Objects in view", span=5, metric_id="all_detections"),
              WidgetDraft(id="total_trend", type="time_series", title="Objects in view, last 5 min", span=12,
                          metric_id="all_detections", window_seconds=300),
              WidgetDraft(id="mix", type="class_distribution", title="By type", span=6),
              WidgetDraft(id="table", type="detection_table", title="Current detections", span=6, rows="detections", max_rows=20)]
        title, kind = "Live counts", "counting"
    draft = DashboardDraft(status="ok", title=title, description=f"Automatic layout for: {task.summary or task.prompt}"[:500],
                           columns=12, metrics=m, widgets=w, limitations=[], clarification_questions=[], rationale="")
    spec, _ = to_spec(draft, caps)
    return spec, kind


# ---------------------------------------------------------------- entry points
def _run(llm: DraftBackend, system: str, messages: List[dict], caps: DashboardCapabilities, max_repairs: int):
    errors: List[str] = []
    for attempt in range(max_repairs + 1):
        draft = llm(system, messages)
        spec, notes = to_spec(draft, caps)
        errors = check_spec(spec, caps)
        if not errors:
            return draft, spec, notes
        log.warning("dashboard draft invalid (attempt %d): %s", attempt + 1, errors)
        messages = messages + [{"role": "assistant", "content": draft.model_dump_json()},
                               {"role": "user", "content": "Invalid:\n- " + "\n- ".join(errors[:12]) + "\nReturn a corrected draft."}]
    raise PlannerUnavailable(f"draft still invalid after {max_repairs + 1} attempts: {errors[:5]}")


def _result_from(draft: DashboardDraft, spec: dict, notes: List[str], llm) -> PlannerResult:
    status = draft.status
    limitations = list(dict.fromkeys(draft.limitations + notes))
    if notes and status == "ok":
        status = "partial"
    return PlannerResult(status=status, spec=spec, ai_generated=True, generator=getattr(llm, "name", "custom"),
                         limitations=limitations, clarification_questions=draft.clarification_questions,
                         rationale=draft.rationale)


def _fallback_result(task, caps, reason: str, extra: Optional[List[str]] = None) -> PlannerResult:
    spec, kind = fallback_spec(task, caps)
    return PlannerResult(
        status="partial", spec=spec, ai_generated=False, generator="fallback:rules", fallback_reason=reason,
        limitations=(extra or []) + ["AI planner unavailable: this is a standard " + kind + " layout built from the "
                                     "task alone; specific requests in your prompt were not interpreted."],
        rationale=f"Deterministic {kind} layout from the TaskSpec.")


def plan_dashboard(prompt: str, task, model_classes=None, sources_catalog: Optional[dict] = None,
                   llm: Optional[DraftBackend] = None, max_repairs: int = 1) -> PlannerResult:
    """Plan a dashboard for the user's goal. Always returns a valid DashboardSpec v1 in `.spec`."""
    caps = build_capabilities(task, model_classes, sources_catalog)
    if task.status != "ok":
        return _fallback_result(task, caps, f"TaskSpec status is {task.status}", task.issues)
    try:
        llm = llm or claude_draft_backend()
        context = {"user_request": prompt[:2000],
                   "task": {"summary": task.summary, "classes": [c.model_dump() for c in task.classes],
                            "alert_class": task.target_event, "primary_metric": task.primary_metric},
                   "capabilities": caps.prompt_block()}
        messages = [{"role": "user", "content": json.dumps(context, separators=(",", ":"))}]
        draft, spec, notes = _run(llm, SYSTEM_PROMPT, messages, caps, max_repairs)
        return _result_from(draft, spec, notes, llm)
    except PlannerUnavailable as e:
        log.warning("dashboard planner falling back: %s", e)
        return _fallback_result(task, caps, str(e))


def refine_dashboard(spec: dict, instruction: str, task, model_classes=None, sources_catalog: Optional[dict] = None,
                     llm: Optional[DraftBackend] = None, max_repairs: int = 1) -> PlannerResult:
    """Apply a follow-up instruction to an existing (valid) DashboardSpec. Never retrains the model."""
    caps = build_capabilities(task, model_classes, sources_catalog)
    if check_spec(spec, caps):
        raise ValueError("current dashboard is not a valid DashboardSpec v1 for this task/model")
    try:
        llm = llm or claude_draft_backend()
        context = {"instruction": instruction[:1000], "current_dashboard": spec, "capabilities": caps.prompt_block()}
        messages = [{"role": "user", "content": json.dumps(context, separators=(",", ":"))}]
        draft, new_spec, notes = _run(llm, REFINE_SYSTEM, messages, caps, max_repairs)
        return _result_from(draft, new_spec, notes, llm)
    except PlannerUnavailable as e:
        log.warning("dashboard refine falling back to rules: %s", e)
        new_spec, applied = rule_refine(spec, instruction, caps)
        return PlannerResult(
            status="ok" if applied else "partial", spec=new_spec, ai_generated=False, generator="fallback:rules",
            fallback_reason=str(e), rationale=applied or "",
            limitations=[] if applied else ["AI planner unavailable and the instruction did not match a simple edit; "
                                            "dashboard unchanged."])


# ---------------------------------------------------------------- rule-based refinement (fallback)
_KIND_WORDS = [
    ("live_preview", r"video|camera|preview|feed|stream|image"),
    ("time_series", r"chart|graph|trend|over time|time series|timeline"),
    ("detection_table", r"table|list|log"),
    ("class_distribution", r"distribution|breakdown|per class|by type|by class"),
    ("kpi", r"kpi|number|counter|count card|tile"),
]


def _kinds_in(text: str) -> List[str]:
    return [k for k, pat in _KIND_WORDS if re.search(pat, text)]


def rule_refine(spec: dict, instruction: str, caps: DashboardCapabilities) -> Tuple[dict, str]:
    """Simple deterministic edits: add / remove / enlarge / shrink widgets. Returns (spec, description or '')."""
    s = json.loads(json.dumps(spec))
    t = instruction.lower()
    kinds = _kinds_in(t)
    cols = s["layout"]["columns"]
    events = "event" in t or "alert" in t

    def matches(w):
        if w["type"] not in kinds:
            return False
        if w["type"] == "detection_table" and ("event" in t or "detection" in t):
            return w["config"]["rows"] == ("events" if events else "detections")
        return True

    if re.search(r"\b(remove|delete|hide|drop|get rid of)\b", t) and kinds:
        before = len(s["widgets"])
        s["widgets"] = [w for w in s["widgets"] if not matches(w)] or s["widgets"]
        removed = before - len(s["widgets"])
        return (s, f"removed {removed} widget(s)") if removed else (spec, "")
    if re.search(r"\b(larger|bigger|wider|enlarge|full width|expand)\b", t) and kinds:
        hit = [w for w in s["widgets"] if matches(w)]
        for w in hit:
            w["span"] = cols if "full" in t else min(cols, w["span"] + max(2, cols // 3))
        return (s, f"enlarged {len(hit)} widget(s)") if hit else (spec, "")
    if re.search(r"\b(smaller|narrower|shrink)\b", t) and kinds:
        hit = [w for w in s["widgets"] if matches(w)]
        for w in hit:
            w["span"] = max(1, w["span"] - max(2, cols // 3))
        return (s, f"shrank {len(hit)} widget(s)") if hit else (spec, "")
    if re.search(r"\b(add|show|include|put)\b", t) and kinds:
        kind = kinds[0]
        cls = next((c for c in caps.class_names if c.replace("_", " ") in t or c in t), None)
        n = len(s["widgets"])
        if kind in ("time_series", "kpi"):
            mid = _slug(f"{cls}_now" if cls else "all_detections", "m")
            if not any(m["id"] == mid for m in s["metrics"]):
                md = {"id": mid, "title": f"{cls.replace('_', ' ').capitalize()} (now)" if cls else "Objects in view",
                      "source_id": SOURCE_ID, "operation": "class_count" if cls else "detection_count"}
                if cls:
                    md["class_name"] = cls
                s["metrics"].append(md)
            label = cls.replace("_", " ").capitalize() if cls else "Detections"
            w = {"id": f"added_{kind}_{n}", "type": kind, "source_id": SOURCE_ID,
                 "title": f"{label} over time" if kind == "time_series" else label,
                 "span": cols if kind == "time_series" else min(cols, 4), "config": {"metric_id": mid}}
            if kind == "time_series":
                w["config"]["window_seconds"] = 300
        elif kind == "detection_table":
            rows = "events" if events and caps.events_available else "detections"
            w = {"id": f"added_table_{n}", "type": kind, "title": "Current " + rows, "source_id": SOURCE_ID,
                 "span": min(cols, 6), "config": {"rows": rows, "max_rows": 20}}
        elif kind == "live_preview":
            w = {"id": f"added_camera_{n}", "type": kind, "title": "Live camera", "source_id": SOURCE_ID,
                 "span": min(cols, 8), "config": {"show_boxes": True}}
        else:
            w = {"id": f"added_mix_{n}", "type": kind, "title": "By type", "source_id": SOURCE_ID,
                 "span": min(cols, 6), "config": {}}
        s["widgets"].append(w)
        return s, f"added {kind}"
    return spec, ""
