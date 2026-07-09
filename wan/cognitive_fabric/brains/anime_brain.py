"""Anime Brain (spec §14) — Phase 1 interface + SeriesBible dataclass.
Anime conditioning compiles into the same prompt/token path as everything
else; style-sheet enforcement and mouth-flap timing are Phase 7 work with
tests already pinning this interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SeriesBible:
    series_id: str
    title: str
    genre: str = ""
    visual_style: Dict[str, Any] = field(default_factory=dict)
    character_bibles: Dict[str, Any] = field(default_factory=dict)
    world_bible: Dict[str, Any] = field(default_factory=dict)
    object_bible: Dict[str, Any] = field(default_factory=dict)
    episode_history: List[str] = field(default_factory=list)
    continuity_rules: List[str] = field(default_factory=list)
    forbidden_changes: List[str] = field(default_factory=list)


class AnimeBrain:
    name = "anime"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "interface",
                "planned": ["style-sheet lock", "mouth-flap timing",
                            "impact frames", "cel-shading validator"]}

    def anime_prompt_clauses(self, style: Dict[str, Any]) -> List[str]:
        clauses = ["anime cinematic style"]
        for key in ("line_weight", "cel_shading", "palette", "background_art"):
            if style.get(key):
                clauses.append(f"{key.replace('_', ' ')}: {style[key]}")
        return clauses
