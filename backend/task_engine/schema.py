"""TaskSpec: the structured contract produced by the Task Engine.

PROPOSED SHARED CONTRACT (v1.0) - needs sign-off from Kolyan (training) and
Damir (evaluation/frontend) before it is treated as frozen.

Design: user intent (what to detect, which event matters, which metric) is kept
separate from `labeling`, the model-specific plan that tells the open-vocabulary
auto-labeller what to look for and how to derive the final training classes.
Open-vocabulary detectors cannot reliably detect negations such as
"person without vest", so those classes are derived by rules from detectable
primitives ("person", "safety vest").
"""
from __future__ import annotations

import re
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SPEC_VERSION = "1.0"
CLASS_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")

TaskType = Literal["object_detection", "image_classification"]
Status = Literal["ok", "needs_clarification", "unsupported"]
Metric = Literal[
    "violation_recall",  # recall on the target_event class
    "recall",
    "precision",
    "f1",
    "map50",
    "accuracy",
]


class ClassDef(BaseModel):
    name: str = Field(description="snake_case training class name, e.g. person_without_vest")
    description: str = Field(description="One sentence describing what a labelled instance looks like")


class DetectorPrompt(BaseModel):
    """A text query sent to the open-vocabulary auto-labeller."""

    key: str = Field(description="snake_case identifier referenced by rules, e.g. person")
    text: str = Field(description="Short positive noun phrase for the detector, e.g. 'person', 'vest'")
    alternatives: List[str] = Field(
        default_factory=list,
        description="2-4 other short phrasings of the same thing; the Data Engine calibrates which one the detector "
                    "responds to best on the user's own frames",
    )
    min_confidence: float = Field(description="Detections below this score are discarded (0.05-0.9)")


class DerivationRule(BaseModel):
    """Turns raw detector outputs into training classes.

    kind="direct":    every `subject` detection becomes `class_if_present`.
    kind="attribute": every `subject` detection is checked for an overlapping
                      `attribute` detection; it becomes `class_if_present` if one
                      overlaps enough, otherwise `class_if_absent`.
    """

    kind: Literal["direct", "attribute"]
    subject: str = Field(description="DetectorPrompt.key whose boxes become labelled objects")
    attribute: Optional[str] = Field(default=None, description="DetectorPrompt.key checked for overlap (attribute rules only)")
    class_if_present: str
    class_if_absent: Optional[str] = Field(default=None, description="Class when the attribute is missing (attribute rules only)")
    min_overlap: float = Field(
        default=0.5,
        description="Fraction of the attribute box that must lie inside the (padded) subject box",
    )
    subject_padding: float = Field(
        default=0.0,
        description="Expand subject box by this fraction of its size before the overlap test (e.g. 0.3 for 'person near table')",
    )


class LabelingPlan(BaseModel):
    detector_prompts: List[DetectorPrompt]
    rules: List[DerivationRule]


class EventRule(BaseModel):
    trigger_class: str = Field(description="Class whose presence triggers the event")
    min_consecutive_frames: int = Field(default=1, description="Debounce for live deployment")
    description: str


class TaskSpecDraft(BaseModel):
    """Fields the LLM fills in. Everything else is added deterministically."""

    status: Status
    issues: List[str] = Field(description="Ambiguities, assumptions or reasons for rejection")
    clarification_questions: List[str]
    task_type: TaskType
    summary: str = Field(description="One-line restatement of the task in plain English")
    classes: List[ClassDef]
    target_event: Optional[str] = Field(description="Class name that represents the alert-worthy situation")
    event_rules: List[EventRule]
    visual_conditions: List[str] = Field(description="Conditions the dataset must cover: lighting, angles, distance, occlusion")
    requires_tracking: bool
    primary_metric: Metric
    secondary_metrics: List[Metric]
    test_scenarios: List[str] = Field(description="Independent real-footage scenarios the evaluation set should contain")
    synthetic_targets: List[str] = Field(description="Under-represented conditions worth generating synthetically (training only)")
    labeling: LabelingPlan


class TaskSpec(TaskSpecDraft):
    spec_version: str = SPEC_VERSION
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    prompt: str
    compiler: str = Field(description="Which engine produced this spec, e.g. claude:claude-opus-5-5")

    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.classes]

    def class_id(self, name: str) -> int:
        return self.class_names.index(name)

    @model_validator(mode="after")
    def _check_consistency(self) -> "TaskSpec":
        errors = consistency_errors(self)
        if errors:
            raise ValueError("; ".join(errors))
        return self


def consistency_errors(d: TaskSpecDraft) -> List[str]:
    """Cross-field checks that JSON Schema alone cannot express."""
    errs: List[str] = []
    if d.status != "ok":
        return errs  # rejected/ambiguous specs are allowed to be partial

    names = [c.name for c in d.classes]
    if not names:
        errs.append("no classes defined")
    if len(set(names)) != len(names):
        errs.append(f"duplicate class names: {names}")
    if len(names) > 10:
        errs.append(f"too many classes for a compact model ({len(names)} > 10)")
    for n in names:
        if not CLASS_NAME_RE.match(n):
            errs.append(f"class name '{n}' must be snake_case")

    if d.task_type == "object_detection":
        keys = [p.key for p in d.labeling.detector_prompts]
        if not keys:
            errs.append("labeling.detector_prompts is empty")
        if len(set(keys)) != len(keys):
            errs.append(f"duplicate detector prompt keys: {keys}")
        for p in d.labeling.detector_prompts:
            if not 0.01 <= p.min_confidence <= 0.95:
                errs.append(f"prompt '{p.key}' min_confidence {p.min_confidence} out of range")
            for t in [p.text, *p.alternatives]:
                if re.search(r"\b(no|not|without|missing)\b", t.lower()):
                    errs.append(f"detector prompt '{t}' contains a negation; derive it with an attribute rule")
        produced = set()
        for r in d.labeling.rules:
            if r.subject not in keys:
                errs.append(f"rule subject '{r.subject}' is not a detector prompt key")
            produced.add(r.class_if_present)
            if r.kind == "attribute":
                if not r.attribute or r.attribute not in keys:
                    errs.append(f"attribute rule needs a valid attribute key, got '{r.attribute}'")
                if not r.class_if_absent:
                    errs.append("attribute rule needs class_if_absent")
                else:
                    produced.add(r.class_if_absent)
                if not 0.0 < r.min_overlap <= 1.0:
                    errs.append(f"min_overlap {r.min_overlap} out of range")
            if not 0.0 <= r.subject_padding <= 2.0:
                errs.append(f"subject_padding {r.subject_padding} out of range")
        unknown = produced - set(names)
        if unknown:
            errs.append(f"rules produce undeclared classes: {sorted(unknown)}")
        missing = set(names) - produced
        if missing:
            errs.append(f"classes never produced by any rule: {sorted(missing)}")

    if d.target_event is not None and d.target_event not in names:
        errs.append(f"target_event '{d.target_event}' is not a declared class")
    for e in d.event_rules:
        if e.trigger_class not in names:
            errs.append(f"event rule trigger '{e.trigger_class}' is not a declared class")
    return errs
