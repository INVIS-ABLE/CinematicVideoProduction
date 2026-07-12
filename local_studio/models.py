"""Pydantic contracts shared by the local studio API and job runner."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    ASPECT_RATIOS,
    MAX_DURATION_SECONDS,
    MAX_IMAGES,
    MAX_VIDEOS,
    MAX_VOICE_SAMPLES,
    MIN_DURATION_SECONDS,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssetKind(str, Enum):
    image = "image"
    video = "video"
    voice = "voice"


class GenerationMode(str, Enum):
    cinematic = "cinematic"
    performance = "performance"
    storyboard = "storyboard"


class JobStatus(str, Enum):
    queued = "queued"
    preparing = "preparing"
    generating = "generating"
    finishing = "finishing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AssetRecord(BaseModel):
    asset_id: str
    project_id: str
    kind: AssetKind
    original_name: str
    relative_path: str
    media_url: str
    content_type: Optional[str] = None
    size_bytes: int = 0
    created_at: str = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectRecord(BaseModel):
    project_id: str
    title: str = "Untitled project"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    assets: List[AssetRecord] = Field(default_factory=list)


class AssetDirective(BaseModel):
    role: str = Field(default="reference", max_length=40)
    note: str = Field(default="", max_length=500)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)


class GenerationRequest(BaseModel):
    project_id: str
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=2000)
    mode: GenerationMode = GenerationMode.cinematic
    duration_seconds: int = Field(default=8, ge=MIN_DURATION_SECONDS, le=MAX_DURATION_SECONDS)
    resolution: str = "720p"
    aspect_ratio: str = "16:9"
    style: str = Field(default="cinematic", max_length=80)
    camera: str = Field(default="director_auto", max_length=80)
    motion: str = Field(default="balanced", max_length=80)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    sampling_steps: int = Field(default=40, ge=8, le=80)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=12.0)
    sample_solver: str = "unipc"
    offload_model: bool = True
    include_audio: bool = True
    image_ids: List[str] = Field(default_factory=list)
    video_ids: List[str] = Field(default_factory=list)
    voice_ids: List[str] = Field(default_factory=list)
    opening_image_id: Optional[str] = None
    motion_video_id: Optional[str] = None
    active_voice_id: Optional[str] = None
    asset_directives: Dict[str, AssetDirective] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("prompt cannot be blank")
        return cleaned

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: str) -> str:
        if value not in {"480p", "720p"}:
            raise ValueError("resolution must be 480p or 720p")
        return value

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect(cls, value: str) -> str:
        if value not in ASPECT_RATIOS:
            raise ValueError(f"unsupported aspect ratio: {value}")
        return value

    @field_validator("sample_solver")
    @classmethod
    def validate_solver(cls, value: str) -> str:
        if value not in {"unipc", "dpm++"}:
            raise ValueError("sample_solver must be unipc or dpm++")
        return value

    @model_validator(mode="after")
    def validate_asset_limits(self) -> "GenerationRequest":
        self.image_ids = list(dict.fromkeys(self.image_ids))
        self.video_ids = list(dict.fromkeys(self.video_ids))
        self.voice_ids = list(dict.fromkeys(self.voice_ids))
        if len(self.image_ids) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images are allowed")
        if len(self.video_ids) > MAX_VIDEOS:
            raise ValueError(f"at most {MAX_VIDEOS} videos are allowed")
        if len(self.voice_ids) > MAX_VOICE_SAMPLES:
            raise ValueError(f"at most {MAX_VOICE_SAMPLES} voice samples are allowed")
        if self.opening_image_id and self.opening_image_id not in self.image_ids:
            raise ValueError("opening_image_id must be included in image_ids")
        if self.motion_video_id and self.motion_video_id not in self.video_ids:
            raise ValueError("motion_video_id must be included in video_ids")
        if self.active_voice_id and self.active_voice_id not in self.voice_ids:
            raise ValueError("active_voice_id must be included in voice_ids")
        return self


class JobRecord(BaseModel):
    job_id: str
    project_id: str
    status: JobStatus = JobStatus.queued
    progress: int = Field(default=0, ge=0, le=100)
    message: str = "Queued"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    request: GenerationRequest
    output_path: Optional[str] = None
    output_url: Optional[str] = None
    error: Optional[str] = None
    report: Dict[str, Any] = Field(default_factory=dict)
