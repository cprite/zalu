"""Planner-side models. The OUTPUT contract is Codex's DashboardSpec v1 (shared/dashboard_spec);
these models are only (a) the flat draft Claude fills in and (b) the result envelope around a valid spec.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

WidgetType = Literal["live_preview", "kpi", "time_series", "detection_table", "class_distribution"]
Operation = Literal["detection_count", "class_count", "inference_latency_ms"]
PlanStatus = Literal["ok", "partial", "needs_clarification", "unsupported"]


class MetricDraft(BaseModel):
    id: str = Field(description="snake_case id, e.g. violations_now")
    title: str = Field(description="Label in the user's own terms, e.g. 'Workers without a vest (now)'")
    operation: Operation
    class_name: Optional[str] = Field(default=None, description="Required for class_count; must be an available class")


class WidgetDraft(BaseModel):
    id: str = Field(description="snake_case id")
    type: WidgetType
    title: str
    span: int = Field(description="Grid columns this widget occupies (1..layout columns)")
    metric_id: Optional[str] = Field(default=None, description="kpi and time_series: id of a metric above")
    window_seconds: Optional[int] = Field(default=None, description="time_series only: 5..600")
    rows: Optional[Literal["detections", "events"]] = Field(default=None, description="detection_table only")
    max_rows: Optional[int] = Field(default=None, description="detection_table only: 1..100")
    show_boxes: Optional[bool] = Field(default=None, description="live_preview only")


class DashboardDraft(BaseModel):
    status: PlanStatus = Field(description="ok: everything requested can be shown; partial: some requests cannot; "
                                           "needs_clarification: unclear what to show; unsupported: nothing requested can be shown")
    title: str
    description: str = Field(description="One sentence: what this dashboard is for")
    columns: int = Field(description="Grid columns, usually 12")
    metrics: List[MetricDraft]
    widgets: List[WidgetDraft] = Field(description="In priority order: the most important information first")
    limitations: List[str] = Field(description="Each requested statistic that cannot be shown, and why")
    clarification_questions: List[str]
    rationale: str = Field(description="One or two sentences on why this layout fits the user's goal")


class PlannerResult(BaseModel):
    status: PlanStatus
    spec: dict = Field(description="Valid DashboardSpec v1 (always present, also for fallback/limited results)")
    ai_generated: bool
    generator: str = Field(description="e.g. claude:claude-opus-5-5 or fallback:rules")
    limitations: List[str] = []
    clarification_questions: List[str] = []
    rationale: str = ""
    fallback_reason: Optional[str] = None
