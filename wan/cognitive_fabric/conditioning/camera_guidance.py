"""Camera guidance: converts a camera plan into precise prompt language and
records the trajectory for continuity checks."""
from __future__ import annotations

from typing import Any, Dict

from ..fabric_types import CognitiveConditioningPack

CAMERA_LANGUAGE = {
    "static": "locked-off static camera",
    "dolly_in": "slow smooth dolly push-in toward the subject",
    "dolly_out": "slow dolly pull-back revealing the scene",
    "orbit": "smooth orbital arc around the subject",
    "crane_up": "rising crane move revealing scale",
    "crane_down": "descending crane move settling on the subject",
    "tracking": "steady lateral tracking shot following the subject",
    "handheld": "subtle handheld sway, documentary energy",
    "low_angle": "low-angle hero framing",
    "reveal": "slow reveal from foreground occlusion",
    "parallax_pan": "lateral pan with strong foreground parallax",
    "over_shoulder": "over-the-shoulder framing",
    "push_in": "gradual push-in increasing intimacy",
}


def inject_camera_guidance(pack: CognitiveConditioningPack,
                           camera_plan: Dict[str, Any]
                           ) -> CognitiveConditioningPack:
    if not camera_plan:
        return pack
    movement = camera_plan.get("movement", "static")
    language = CAMERA_LANGUAGE.get(movement, movement)
    lens = camera_plan.get("lens")
    clause = f"camera: {language}"
    if lens:
        clause += f", {lens} lens"
    if camera_plan.get("easing"):
        clause += f", {camera_plan['easing']} easing"
    pack.prompt_clauses.append(clause)
    pack.negative_clauses.extend(["chaotic camera movement", "sudden camera jumps"])
    pack.shot_metadata["camera_plan"] = dict(camera_plan)
    return pack
