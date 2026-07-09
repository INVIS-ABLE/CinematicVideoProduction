"""Object Brain: registers storyboard objects and produces continuity
conditioning for props/vehicles/etc."""
from __future__ import annotations

from typing import Any, Dict, List

from ..memory.object_memory import ObjectMemory


class ObjectBrain:
    name = "object"
    kind = "brain"

    def __init__(self, memory: ObjectMemory):
        self.memory = memory

    def health_check(self) -> Dict[str, Any]:
        return self.memory.health_check()

    def register_objects(self, storyboard: Dict[str, Any]) -> List[str]:
        registered = []
        for obj in storyboard.get("objects", []):
            self.memory.register(
                obj["object_id"], obj.get("name", obj["object_id"]),
                object_type=obj.get("type", "prop"),
                material=obj.get("material", ""),
                colour=obj.get("colour", ""),
                damage_state=obj.get("damage_state", "intact"))
            registered.append(obj["object_id"])
        return registered
