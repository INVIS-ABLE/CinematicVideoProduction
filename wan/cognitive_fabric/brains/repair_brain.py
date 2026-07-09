"""Repair Brain (spec §19): maps a quality report + attempt count to a
concrete repair action, capped by config. Every decision is logged to the
memory DB by the repair controller."""
from __future__ import annotations

from typing import Any, Dict

from ..fabric_types import QualityReport

# ordered from cheapest to most invasive
ACTIONS = [
    "accept",
    "accept_with_warning",
    "stabilise",
    "detail_pass",
    "adjust_seed",
    "strengthen_identity_tokens",
    "strengthen_world_tokens",
    "reduce_motion_complexity",
    "reduce_camera_complexity",
    "use_previous_frame_conditioning",
    "split_shot",
    "regenerate",
    "flag_for_user",
]


class RepairBrain:
    name = "repair"
    kind = "brain"

    def __init__(self, max_attempts: int = 3,
                 thresholds: Dict[str, float] = None):
        self.max_attempts = max_attempts
        self.thresholds = thresholds or {}

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "max_attempts": self.max_attempts}

    def decide(self, report: QualityReport, attempt: int) -> Dict[str, Any]:
        """Return {action, reason, exhausted}."""
        if attempt >= self.max_attempts:
            return {"action": "accept_with_warning", "exhausted": True,
                    "reason": f"repair budget exhausted after {attempt} attempts"}

        recommended = report.recommended_action
        if recommended == "accept":
            return {"action": "accept", "exhausted": False, "reason": "passed"}

        if recommended == "regenerate":
            # escalate conditioning strength before burning a raw retry
            action = ("use_previous_frame_conditioning" if attempt == 0 else
                      "strengthen_identity_tokens" if attempt == 1 else
                      "regenerate")
            return {"action": action, "exhausted": False,
                    "reason": "identity/world drift detected"}

        if recommended in ("stabilise", "detail_pass"):
            return {"action": recommended, "exhausted": False,
                    "reason": f"quality brain recommends {recommended}"}

        return {"action": "adjust_seed", "exhausted": False,
                "reason": f"unrecognised recommendation {recommended!r}"}
