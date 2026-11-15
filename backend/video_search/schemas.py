"""Video search data contracts (P0). Times are milliseconds from the start of the recording, derived from the
GX10 worker's per-frame t_s = frame_index / container fps (not PTS; see STATUS.md)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.schemas import new_id, now

MAX_RECORDINGS = 10
EMBEDDING_VERSION = "clip-ViT-B/32:crop-pad0.1:v1"
SCORE_TYPE = "clip_cosine_similarity"

RecState = Literal["created", "uploaded", "queued", "processing", "completed", "failed"]


class RecordingIn(BaseModel):
    camera_id: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=255)
    recording_start_utc: Optional[datetime] = None

    @field_validator("recording_start_utc")
    @classmethod
    def _tz(cls, v):
        if v is not None and v.tzinfo is None:
            raise ValueError("recording_start_utc must include a timezone (e.g. ...Z)")
        return v


class BatchCreate(BaseModel):
    name: str = "video search batch"
    recordings: list[RecordingIn] = Field(min_length=1)
    fps: float = Field(2.0, ge=0.1, le=10)
    conf: float = Field(0.25, ge=0.01, le=0.99)

    @field_validator("recordings")
    @classmethod
    def _max(cls, v):
        if len(v) > MAX_RECORDINGS:
            raise ValueError(f"a batch holds at most {MAX_RECORDINGS} recordings, got {len(v)}")
        return v


class Recording(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rec"))
    camera_id: str
    filename: str
    recording_start_utc: Optional[datetime] = None
    state: RecState = "created"
    stage: Optional[str] = None  # gx10_upload / detection / tracking / embedding / done
    error: Optional[str] = None
    size_bytes: Optional[int] = None
    probe: dict = Field(default_factory=dict)  # ffprobe: duration_s, fps, width, height (null if unknown)
    gx10_path: Optional[str] = None  # internal, never returned by the API
    sampled_frames: Optional[int] = None
    frames_with_person: Optional[int] = None
    detections: Optional[int] = None
    tracklets: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    timings_s: dict = Field(default_factory=dict)  # measured wall time per stage


class Batch(BaseModel):
    id: str = Field(default_factory=lambda: new_id("batch"))
    name: str
    fps: float = 2.0
    conf: float = 0.25
    status: Literal["created", "queued", "processing", "completed", "partial", "failed"] = "created"
    model_id: Optional[str] = None  # zero-shot person artifact, created once per batch
    recordings: list[Recording]
    created_at: float = Field(default_factory=now)
    queued_at: Optional[float] = None
    finished_at: Optional[float] = None


class Tracklet(BaseModel):
    tracklet_id: str
    batch_id: str
    recording_id: str
    camera_id: str
    start_ms: int
    end_ms: int
    detection_count: int
    max_score: float
    rep_frame_index: int
    rep_t_ms: int
    rep_box: list[float]
    embedding_version: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    camera_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(50, ge=1, le=500)
    min_score: Optional[float] = None
