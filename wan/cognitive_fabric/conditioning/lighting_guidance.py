"""Lighting guidance: converts a lighting plan into prompt clauses and locks
light continuity fields into scene metadata."""
from __future__ import annotations

from typing import Any, Dict

from ..fabric_types import CognitiveConditioningPack


def inject_lighting_guidance(pack: CognitiveConditioningPack,
                             lighting_plan: Dict[str, Any]
                             ) -> CognitiveConditioningPack:
    if not lighting_plan:
        return pack
    parts = []
    for key in ("key", "fill", "rim", "practicals", "mood", "colour_temperature"):
        if lighting_plan.get(key):
            parts.append(f"{key.replace('_', ' ')}: {lighting_plan[key]}")
    if parts:
        pack.prompt_clauses.append("cinematic lighting — " + "; ".join(parts))
    if lighting_plan.get("direction"):
        pack.prompt_clauses.append(
            f"light direction stays {lighting_plan['direction']} across the shot")
        pack.scene_metadata["light_direction"] = lighting_plan["direction"]
    pack.negative_clauses.extend([
        "inconsistent lighting", "shadows changing direction", "flat lighting",
    ])
    return pack
