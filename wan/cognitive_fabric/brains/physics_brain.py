"""Physics Brain: builds physics expectations per shot and hands violations
to the Repair Brain. Detection beyond pixel statistics is a documented
Phase-later upgrade (see validators/physics_validator.py)."""
from __future__ import annotations

from typing import Any, Dict

from ..conditioning.physics_guidance import build_physics_expectation
from ..fabric_types import PhysicsExpectation


class PhysicsBrain:
    name = "physics"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "expectation+prompt-guidance",
                "planned": ["optical-flow validation", "collision masks"]}

    def expectation_for_shot(self, shot: Dict[str, Any]) -> PhysicsExpectation:
        return build_physics_expectation(
            shot.get("prompt", ""), shot.get("motion_plan"))
