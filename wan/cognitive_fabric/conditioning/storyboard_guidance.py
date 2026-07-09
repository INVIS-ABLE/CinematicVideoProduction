"""Storyboard-aware MoE routing (spec §11).

Wan 2.2's A14B "MoE" is expert-per-noise-band: the high-noise expert owns
layout/composition, the low-noise expert owns detail. This module maps shot
metadata to CognitiveMoERoutingState priorities. Phase 1 consumes it as
guide-scale shaping and prompt emphasis (real, testable influence on the
generation call); deeper per-expert conditioning is Phase 4.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..fabric_types import CognitiveMoERoutingState

_SHOT_TYPE_LAYOUT_WEIGHT = {
    "establishing": 0.9, "wide": 0.85, "aerial": 0.9, "action": 0.8,
    "medium": 0.6, "tracking": 0.7,
    "close-up": 0.3, "insert": 0.25, "reaction": 0.35, "macro": 0.2,
}


def build_routing_state(shot_metadata: Dict[str, Any]) -> CognitiveMoERoutingState:
    shot_type = str(shot_metadata.get("shot_type", "medium")).lower()
    layout = 0.5
    for key, weight in _SHOT_TYPE_LAYOUT_WEIGHT.items():
        if key in shot_type:
            layout = weight
            break
    state = CognitiveMoERoutingState(
        layout_priority=layout,
        detail_priority=1.0 - layout * 0.7,
        identity_priority=0.9 if shot_metadata.get("character_ids") else 0.3,
        physics_priority=0.8 if shot_metadata.get("physics_expectation") else 0.4,
        world_priority=0.8 if shot_metadata.get("world_id") else 0.4,
        camera_priority=0.7 if shot_metadata.get("camera_plan") else 0.3,
        lighting_priority=0.7 if shot_metadata.get("lighting_plan") else 0.4,
        anime_priority=1.0 if shot_metadata.get("anime_mode") else 0.0,
        texture_priority=0.6,
    )
    return state.clamp()


def shape_guide_scales(state: CognitiveMoERoutingState,
                       base_low: float, base_high: float,
                       max_delta: float = 0.5) -> Tuple[float, float]:
    """Shape CFG per expert: layout-priority shots push the high-noise
    (layout) expert's guidance up slightly; detail shots push the low-noise
    expert. Bounded by max_delta so it can never destabilise sampling."""
    high = base_high + max_delta * (state.layout_priority - 0.5) * 2
    low = base_low + max_delta * (state.detail_priority - 0.5) * 2
    return (round(max(1.0, low), 3), round(max(1.0, high), 3))
