"""Shared data contracts. TaskSpec / Annotation / DatasetManifest come from Kirill's engines (agreed);
TrainingJob / ModelArtifact / InferenceResult are Kolyan's; EvaluationReport is a placeholder for Damir.

Every id is a short string with a type prefix: task_, ds_, job_, model_, rep_, inf_.
Boxes are in pixels of the original image: [x1, y1, x2, y2].
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now() -> float:
    return round(time.time(), 3)


# ---------- Task Engine + Data Engine (Kirill) ----------
# AGREED 2026-11-14: Kirill's contracts are the shared source of truth for these objects.
# The backend re-exports them; do not redefine them here.
from backend.task_engine.schema import TaskSpec  # noqa: E402,F401  (id field: task_id)
from backend.data_engine.schema import (  # noqa: E402,F401  (id field: dataset_id)
    Annotation, AnnotationSet, BoxLabel, DatasetManifest, LabelCorrection, SplitStats,
)


def obj_id(obj: BaseModel) -> str:
    """Storage key for any shared object (Kirill's use task_id / dataset_id, the backend's use id)."""
    for f in ("id", "dataset_id", "task_id", "annotation_set_id"):
        v = getattr(obj, f, None)
        if v:
            return v
    raise ValueError(f"{type(obj).__name__} has no id field")


# ---------- Training Engine (Kolyan) ----------
class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class TrainingRequest(BaseModel):
    dataset_id: str
    base_model: str = "yolo11n.pt"            # compact by default
    epochs: int = Field(30, ge=1, le=300)
    imgsz: int = Field(640, ge=320, le=1920)
    batch: int = Field(16, ge=1, le=128)
    export_onnx: bool = True


class TrainingJob(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    kind: Literal["training", "evaluation"] = "training"
    status: JobStatus = JobStatus.queued
    request: dict[str, Any] = {}
    progress: float = 0.0                     # 0..1
    message: str = ""
    error: Optional[str] = None
    result_id: Optional[str] = None           # model_ id for training jobs
    created_at: float = Field(default_factory=now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class ModelArtifact(BaseModel):
    id: str = Field(default_factory=lambda: new_id("model"))
    kind: Literal["trained", "zero_shot"] = "trained"
    dataset_id: Optional[str] = None
    variant: Optional[str] = None
    base_model: str
    classes: list[str]
    weights: Optional[str] = None             # .pt path
    onnx: Optional[str] = None
    metrics: dict[str, float] = {}            # on the dataset's val split: precision, recall, map50, map50_95
    baseline_metrics: dict[str, float] = {}   # base model on the same val split BEFORE fine-tuning (if classes match)
    export_check: dict[str, Any] = {}         # ONNX vs .pt on one val image: detections, latency, ok
    train_seconds: Optional[float] = None
    epochs: Optional[int] = None
    imgsz: Optional[int] = None
    device: Optional[str] = None
    created_at: float = Field(default_factory=now)


# ---------- Evaluation Engine (Damir) ----------
class EvaluationRequest(BaseModel):
    model_ids: list[str]                      # e.g. [zero-shot baseline, real-only, real+synthetic]
    test_dataset_id: str                      # independent real recordings, never used for training


class EvaluationReport(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rep"))
    request: EvaluationRequest
    per_model: dict[str, dict[str, Any]] = {} # model_id -> {precision, recall, f1, fp, fn, fps, ...}
    notes: list[str] = []
    mock: bool = False                        # True until Damir's Evaluation Engine is merged
    created_at: float = Field(default_factory=now)


# ---------- Deployment Engine (Kolyan) ----------
class Detection(BaseModel):
    cls: str
    score: float
    box: list[float]


class InferenceResult(BaseModel):
    id: str = Field(default_factory=lambda: new_id("inf"))
    model_id: str
    width: int
    height: int
    detections: list[Detection]
    latency_ms: float
    events: list[str] = []                    # e.g. ["person_without_vest"] when target_event fires
