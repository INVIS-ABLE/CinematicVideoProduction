"""World Brain: extracts and maintains world state from storyboards."""
from __future__ import annotations

from typing import Any, Dict, List

from ..memory.world_memory import WorldMemory, WorldState


class WorldBrain:
    name = "world"
    kind = "brain"

    def __init__(self, memory: WorldMemory):
        self.memory = memory

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "worlds": len(self.memory.worlds)}

    def register_worlds(self, storyboard: Dict[str, Any]) -> List[WorldState]:
        registered = []
        for world in storyboard.get("worlds", []):
            kwargs = {k: v for k, v in world.items() if k != "world_id"}
            allowed = {f for f in WorldState.__dataclass_fields__}
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            registered.append(
                self.memory.create_world(world["world_id"], **kwargs))
        return registered
