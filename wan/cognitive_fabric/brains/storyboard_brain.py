"""Storyboard Brain: validates and normalises storyboard documents against
the schema contract used by the orchestrator."""
from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_SHOT_FIELDS = ("shot_id", "scene_id", "duration_seconds", "prompt")


class StoryboardBrain:
    name = "storyboard"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def validate(self, storyboard: Dict[str, Any]) -> List[str]:
        """Returns a list of problems; empty list = valid."""
        problems: List[str] = []
        if "project" not in storyboard:
            problems.append("missing 'project' section")
        shots = storyboard.get("shots", [])
        if not shots:
            problems.append("storyboard has no shots")
        seen_ids = set()
        scene_ids = {s.get("scene_id") for s in storyboard.get("scenes", [])}
        for shot in shots:
            sid = shot.get("shot_id", "<missing>")
            for field in REQUIRED_SHOT_FIELDS:
                if field not in shot:
                    problems.append(f"{sid}: missing field '{field}'")
            if sid in seen_ids:
                problems.append(f"duplicate shot_id {sid}")
            seen_ids.add(sid)
            if shot.get("duration_seconds", 0) <= 0:
                problems.append(f"{sid}: duration_seconds must be > 0")
            if scene_ids and shot.get("scene_id") not in scene_ids:
                problems.append(f"{sid}: unknown scene_id {shot.get('scene_id')}")
        return problems

    def normalise(self, storyboard: Dict[str, Any]) -> Dict[str, Any]:
        """Fill optional fields with defaults, in place-safe copy."""
        import copy
        sb = copy.deepcopy(storyboard)
        sb.setdefault("characters", [])
        sb.setdefault("objects", [])
        sb.setdefault("worlds", [])
        sb.setdefault("scenes", [])
        sb.setdefault("acts", [])
        for shot in sb.get("shots", []):
            shot.setdefault("negative_prompt", "")
            shot.setdefault("character_ids", [])
            shot.setdefault("object_ids", [])
            shot.setdefault("world_id", None)
            shot.setdefault("camera_plan", {})
            shot.setdefault("lighting_plan", {})
            shot.setdefault("motion_plan", {})
            shot.setdefault("physics_expectation", {})
            shot.setdefault("transition_in", "cut")
            shot.setdefault("transition_out", "cut")
            shot.setdefault("continuity_from_previous", True)
            shot.setdefault("continuity_to_next", True)
            shot.setdefault("seed_strategy", "per_shot")
        return sb
