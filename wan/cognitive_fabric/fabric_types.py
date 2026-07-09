"""Core datatypes shared across the cognitive fabric.

Tensor-carrying types keep tensors optional so planning/dry-run paths work
without allocating anything; `validate()` enforces the shape/dtype contracts
documented in fable_memory/wan22_tensor_shape_map.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


# --------------------------------------------------------------------------
# Reference conditioning
# --------------------------------------------------------------------------

REFERENCE_ROLES = (
    "image", "video", "audio", "storyboard_frame", "character_sheet",
    "object_sheet", "world", "style", "previous_frame", "previous_clip",
    "approved_character_frame", "approved_object_frame", "lighting", "camera",
)


@dataclass
class ReferenceToken:
    """One conditioning reference compiled into the token stack."""

    id: str
    role: str
    source_path: Optional[str] = None
    embedding: Optional[torch.Tensor] = None  # [D] or [n, D]
    weight: float = 1.0
    priority: int = 5  # 0 (drop first) .. 10 (never drop)
    scene_scope: Optional[str] = None
    shot_scope: Optional[str] = None
    character_scope: Optional[str] = None
    object_scope: Optional[str] = None
    temporal_validity: Optional[str] = None  # e.g. "scene", "shot", "project"
    persist: bool = False

    def validate(self) -> None:
        if self.role not in REFERENCE_ROLES:
            raise ValueError(f"unknown reference role: {self.role!r}")
        if self.embedding is not None and self.embedding.dim() not in (1, 2):
            raise ValueError(
                f"reference embedding must be [D] or [n, D], got "
                f"{tuple(self.embedding.shape)}")


# --------------------------------------------------------------------------
# Conditioning pack (spec §5) — the single object handed to Wan hooks
# --------------------------------------------------------------------------


@dataclass
class CognitiveConditioningPack:
    """Everything the fabric wants Wan's generation step to know.

    Phase 1: consumed by the prompt compiler and generation controller
    (text-level conditioning + metadata). Phase 4 wires the token tensors
    into cross-attention context and the bias tensors into self-attention.
    """

    text_tokens: Optional[torch.Tensor] = None          # [B, L_txt, C_txt]
    reference_tokens: Optional[torch.Tensor] = None     # [n_ref, C_txt]
    identity_tokens: Optional[torch.Tensor] = None      # [n_id, C_txt]
    object_tokens: Optional[torch.Tensor] = None        # [n_obj, C_txt]
    world_tokens: Optional[torch.Tensor] = None         # [n_world, C_txt]
    storyboard_tokens: Optional[torch.Tensor] = None    # [n_sb, C_txt]
    physics_tokens: Optional[torch.Tensor] = None       # [n_phys, C_txt]
    lighting_tokens: Optional[torch.Tensor] = None      # [n_light, C_txt]
    camera_tokens: Optional[torch.Tensor] = None        # [n_cam, C_txt]
    audio_tokens: Optional[torch.Tensor] = None         # [n_aud, C_txt]
    temporal_cache: Optional[Any] = None                # TemporalKVCache
    attention_bias: Optional[torch.Tensor] = None       # [L, L] additive
    collision_bias: Optional[torch.Tensor] = None       # [L, L] additive
    timestep_metadata: Dict[str, Any] = field(default_factory=dict)
    scene_metadata: Dict[str, Any] = field(default_factory=dict)
    shot_metadata: Dict[str, Any] = field(default_factory=dict)
    # text-level conditioning consumed today (Phase 1):
    prompt_clauses: List[str] = field(default_factory=list)
    negative_clauses: List[str] = field(default_factory=list)
    reference_manifest: List[Dict[str, Any]] = field(default_factory=list)

    _TOKEN_FIELDS = (
        "reference_tokens", "identity_tokens", "object_tokens", "world_tokens",
        "storyboard_tokens", "physics_tokens", "lighting_tokens",
        "camera_tokens", "audio_tokens",
    )

    def validate(self) -> None:
        dims = set()
        for name in self._TOKEN_FIELDS:
            t = getattr(self, name)
            if t is None:
                continue
            if t.dim() != 2:
                raise ValueError(f"{name} must be [n, C], got {tuple(t.shape)}")
            dims.add(t.shape[-1])
        if len(dims) > 1:
            raise ValueError(f"token channel dims disagree: {sorted(dims)}")
        for name in ("attention_bias", "collision_bias"):
            t = getattr(self, name)
            if t is not None and t.dim() != 2:
                raise ValueError(f"{name} must be [L, L], got {tuple(t.shape)}")

    def token_count(self) -> int:
        return sum(
            getattr(self, n).shape[0]
            for n in self._TOKEN_FIELDS if getattr(self, n) is not None)

    def summary(self) -> Dict[str, Any]:
        return {
            "token_count": self.token_count(),
            "has_temporal_cache": self.temporal_cache is not None,
            "has_attention_bias": self.attention_bias is not None,
            "prompt_clauses": len(self.prompt_clauses),
            "negative_clauses": len(self.negative_clauses),
            "references_used": len(self.reference_manifest),
            "shot": self.shot_metadata.get("shot_id"),
            "scene": self.scene_metadata.get("scene_id"),
        }


# --------------------------------------------------------------------------
# MoE routing metadata (spec §11) — conditioning influence, honestly scoped
# --------------------------------------------------------------------------


@dataclass
class CognitiveMoERoutingState:
    """Task-aware priorities for the high-noise (layout) vs low-noise
    (detail) experts. Phase 1 consumes this as prompt-emphasis and
    guide-scale shaping; deeper model-level use lands in Phase 4."""

    layout_priority: float = 0.5
    detail_priority: float = 0.5
    identity_priority: float = 0.5
    physics_priority: float = 0.5
    world_priority: float = 0.5
    camera_priority: float = 0.5
    lighting_priority: float = 0.5
    anime_priority: float = 0.0
    texture_priority: float = 0.5
    repair_focus_mask: Optional[torch.Tensor] = None

    def clamp(self) -> "CognitiveMoERoutingState":
        for name in ("layout_priority", "detail_priority", "identity_priority",
                     "physics_priority", "world_priority", "camera_priority",
                     "lighting_priority", "anime_priority", "texture_priority"):
            setattr(self, name, min(1.0, max(0.0, getattr(self, name))))
        return self


# --------------------------------------------------------------------------
# Physics expectation (spec §12)
# --------------------------------------------------------------------------


@dataclass
class PhysicsExpectation:
    subjects: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    expected_contacts: List[str] = field(default_factory=list)
    expected_collisions: List[str] = field(default_factory=list)
    expected_motion_paths: List[str] = field(default_factory=list)
    gravity_direction: str = "down"
    floor_plane_estimate: Dict[str, Any] = field(default_factory=dict)
    forbidden_intersections: List[str] = field(default_factory=list)
    material_behaviours: Dict[str, str] = field(default_factory=dict)
    particle_behaviours: Dict[str, str] = field(default_factory=dict)

    def to_prompt_clauses(self) -> List[str]:
        clauses = []
        if self.expected_contacts:
            clauses.append("grounded physical contact: " +
                           ", ".join(self.expected_contacts))
        for material, behaviour in self.material_behaviours.items():
            clauses.append(f"{material} behaves realistically: {behaviour}")
        return clauses


# --------------------------------------------------------------------------
# Storyboard shot / chunk specs (spec §9, §15)
# --------------------------------------------------------------------------


@dataclass
class ShotSpec:
    shot_id: str
    scene_id: str
    duration_seconds: float
    prompt: str
    negative_prompt: str = ""
    character_ids: List[str] = field(default_factory=list)
    object_ids: List[str] = field(default_factory=list)
    world_id: Optional[str] = None
    camera_plan: Dict[str, Any] = field(default_factory=dict)
    lighting_plan: Dict[str, Any] = field(default_factory=dict)
    motion_plan: Dict[str, Any] = field(default_factory=dict)
    physics_expectation: Dict[str, Any] = field(default_factory=dict)
    transition_in: str = "cut"
    transition_out: str = "cut"
    audio_reference: Optional[str] = None
    style_reference: Optional[str] = None
    continuity_from_previous: bool = True
    continuity_to_next: bool = True
    seed_strategy: str = "per_shot"
    output_path: Optional[str] = None

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError(f"{self.shot_id}: duration must be > 0")
        if not self.prompt.strip():
            raise ValueError(f"{self.shot_id}: prompt is empty")


@dataclass
class ChunkSpec:
    """One Wan generation call. frame_num obeys the 4n+1 invariant."""

    chunk_id: str
    shot_id: str
    scene_id: str
    index: int
    frame_num: int
    fps: int
    width: int
    height: int
    seed: int
    prompt: str
    negative_prompt: str
    first_frame_path: Optional[str] = None   # continuity via TI2V i2v clamp
    overlap_frames: int = 0
    status: str = "planned"  # planned|generating|validated|repaired|failed|approved
    attempt: int = 0
    output_path: Optional[str] = None

    def validate(self) -> None:
        if (self.frame_num - 1) % 4 != 0:
            raise ValueError(
                f"{self.chunk_id}: frame_num must be 4n+1, got {self.frame_num}")
        if self.width % 16 or self.height % 16:
            raise ValueError(
                f"{self.chunk_id}: width/height must be multiples of 16")

    def duration_seconds(self) -> float:
        return self.frame_num / max(self.fps, 1)


# --------------------------------------------------------------------------
# Quality report (spec §19)
# --------------------------------------------------------------------------


@dataclass
class QualityReport:
    chunk_id: str
    shot_id: str
    scene_id: str
    prompt_match: float = 0.0
    identity_score: float = 0.0
    object_score: float = 0.0
    world_score: float = 0.0
    motion_score: float = 0.0
    physics_score: float = 0.0
    lighting_score: float = 0.0
    anatomy_score: float = 0.0
    sharpness_score: float = 0.0
    cinematic_score: float = 0.0
    flicker_score: float = 0.0
    failure_regions: List[Dict[str, Any]] = field(default_factory=list)
    recommended_action: str = "accept"
    measured: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def overall(self) -> float:
        scores = [
            self.prompt_match, self.identity_score, self.object_score,
            self.world_score, self.motion_score, self.physics_score,
            self.lighting_score, self.anatomy_score, self.sharpness_score,
            self.cinematic_score,
        ]
        active = [s for s in scores if s > 0]
        return sum(active) / len(active) if active else 0.0

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["overall"] = self.overall()
        return d
