"""Data Engine contracts: Frame, FrameCollection, Annotation, DatasetManifest.

PROPOSED SHARED CONTRACT (v1.0) - Annotation/BoxLabel are what the frontend
review screen reads and writes back (Damir); DatasetManifest is what the
Training Engine consumes (Kolyan). Needs team sign-off.

Conventions:
- Boxes are absolute pixel xyxy in the stored image (frontend-friendly);
  YOLO normalised xywh only exists inside the exported label files.
- session_id identifies one independent recording. Splits are made per session.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
Split = Literal["train", "val", "test"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Frame(BaseModel):
    frame_id: str
    path: str = Field(description="Absolute path of the extracted JPEG")
    session_id: str = Field(description="Independent recording session this frame came from")
    source_file: str
    source_frame_index: Optional[int] = None
    timestamp_s: Optional[float] = None
    width: int
    height: int
    sharpness: float = Field(description="Variance of Laplacian; higher is sharper")
    dhash: str = Field(description="64-bit difference hash, hex")
    synthetic: bool = False
    parent_frame_id: Optional[str] = Field(default=None, description="Real frame a synthetic image was derived from")
    generation: Optional[dict] = Field(default=None, description="Synthetic provenance: model, prompt, seed, settings")


class RejectedFrame(BaseModel):
    source_file: str
    source_frame_index: Optional[int] = None
    reason: Literal["unreadable", "blurry", "near_duplicate", "too_small", "not_selected"]
    detail: str = ""


class FrameCollection(BaseModel):
    schema_version: str = SCHEMA_VERSION
    collection_id: str = Field(default_factory=lambda: new_id("frames"))
    created_at: str = Field(default_factory=now_iso)
    root: str
    frames: List[Frame]
    rejected: List[RejectedFrame] = []
    sessions: Dict[str, List[str]] = Field(default_factory=dict, description="session_id -> source files")
    settings: dict = {}


class BoxLabel(BaseModel):
    class_name: str
    xyxy: List[float] = Field(description="[x1, y1, x2, y2] absolute pixels")
    confidence: Optional[float] = Field(default=None, description="None for manual boxes")
    source: Literal["auto", "manual"] = "auto"
    evidence: Optional[dict] = Field(default=None, description="Raw detections that produced this box (subject/attribute scores)")


class Annotation(BaseModel):
    frame_id: str
    image_path: str
    width: int
    height: int
    session_id: str
    synthetic: bool = False
    boxes: List[BoxLabel] = []
    status: Literal["auto", "reviewed", "rejected"] = "auto"
    needs_review: bool = False
    review_reasons: List[str] = []


class AnnotationSet(BaseModel):
    schema_version: str = SCHEMA_VERSION
    annotation_set_id: str = Field(default_factory=lambda: new_id("ann"))
    created_at: str = Field(default_factory=now_iso)
    task_id: str
    class_names: List[str]
    annotator: str = Field(description="e.g. yolo-world:yolov8s-worldv2.pt")
    annotations: List[Annotation]
    stats: dict = {}


class LabelCorrection(BaseModel):
    """Body of POST /api/datasets/{id}/labels - one entry per edited frame.

    `boxes` replaces the frame's boxes entirely (simplest contract for a UI editor).
    `status="rejected"` drops the frame from the export.
    """

    frame_id: str
    boxes: Optional[List[BoxLabel]] = None
    status: Literal["reviewed", "rejected"] = "reviewed"


class SplitStats(BaseModel):
    images: int = 0
    background_images: int = 0
    boxes: int = 0
    per_class: Dict[str, int] = {}
    sessions: List[str] = []
    synthetic_images: int = 0


class ManifestImage(BaseModel):
    frame_id: str
    split: Split
    image: str = Field(description="Path relative to dataset root")
    label: str
    session_id: str
    source_file: str
    synthetic: bool
    parent_frame_id: Optional[str] = None
    review_status: str


class DatasetManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    dataset_id: str = Field(default_factory=lambda: new_id("ds"))
    created_at: str = Field(default_factory=now_iso)
    task_id: str
    variant: Literal["real_only", "real_plus_synthetic", "test"]
    root: str = Field(description="Absolute dataset directory")
    data_yaml: str = Field(description="Absolute path to data.yaml (pass to `yolo train data=...`)")
    class_names: List[str] = Field(description="Index = YOLO class id")
    splits: Dict[str, SplitStats]
    split_strategy: str
    excluded_sessions: List[str] = Field(default=[], description="Held-out sessions guaranteed absent from this dataset")
    images: List[ManifestImage]
    warnings: List[str] = []
    annotator: Optional[str] = None
    timings: Dict[str, float] = Field(default={}, description="Seconds per Data Engine stage (measured, wall clock)")
