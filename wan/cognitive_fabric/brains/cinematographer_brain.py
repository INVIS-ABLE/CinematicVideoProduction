"""Cinematographer Brain: professional camera language per shot, kept
compatible with the scene (no chaotic movement unless asked)."""
from __future__ import annotations

from typing import Any, Dict, Optional

_SHOT_TYPE_CAMERA = {
    "establishing wide": ("crane_down", "24mm", "slow"),
    "extreme wide": ("static", "18mm", "none"),
    "medium tracking": ("tracking", "35mm", "steady"),
    "medium": ("static", "35mm", "none"),
    "close-up": ("push_in", "85mm", "very slow"),
    "extreme close-up": ("static", "100mm macro", "none"),
    "insert detail": ("static", "50mm", "none"),
    "action": ("handheld", "35mm", "energetic"),
    "reaction": ("static", "85mm", "none"),
    "final hero shot": ("dolly_in", "35mm", "slow"),
    "over-the-shoulder": ("over_shoulder", "50mm", "none"),
    "aerial": ("crane_up", "24mm", "slow"),
}


class CinematographerBrain:
    name = "cinematographer"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "vocabulary": len(_SHOT_TYPE_CAMERA)}

    def plan_camera(self, shot_type: str,
                    user_plan: Optional[Dict[str, Any]] = None,
                    motion_intensity: str = "moderate") -> Dict[str, Any]:
        movement, lens, speed = _SHOT_TYPE_CAMERA.get(
            shot_type.lower(), ("static", "35mm", "none"))
        plan = {
            "movement": movement, "lens": lens, "speed": speed,
            "easing": "ease-in-out", "subject_lock": True,
            "depth_of_field": ("shallow" if "close" in shot_type.lower()
                               else "deep"),
            "framing": ("rule of thirds" if "hero" not in shot_type.lower()
                        else "centered hero framing"),
        }
        if motion_intensity == "calm":
            plan["speed"] = "very slow"
        if user_plan:
            plan.update({k: v for k, v in user_plan.items() if v})
        return plan
