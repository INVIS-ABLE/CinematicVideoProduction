"""Inject world-state tokens/clauses into a conditioning pack."""
from __future__ import annotations

from typing import Optional

from ..fabric_types import CognitiveConditioningPack
from ..memory.world_memory import WorldMemory


def inject_world_tokens(pack: CognitiveConditioningPack,
                        worlds: WorldMemory,
                        world_id: Optional[str]) -> CognitiveConditioningPack:
    if world_id:
        pack.prompt_clauses.extend(worlds.get_world_conditioning(world_id))
        state = worlds.get(world_id)
        if state is not None:
            pack.scene_metadata.setdefault("world_id", world_id)
            pack.scene_metadata.setdefault("time_of_day", state.time_of_day)
            pack.scene_metadata.setdefault("weather", state.weather)
    pack.validate()
    return pack
