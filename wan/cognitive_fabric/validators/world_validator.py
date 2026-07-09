"""World validator: delegates drift detection to WorldMemory field
comparison; VLM-observed world description arrives in a later phase."""
from __future__ import annotations

from typing import Any, Dict, List

from ..memory.world_memory import WorldMemory


def check_world(worlds: WorldMemory, world_id: str,
                observed: Dict[str, Any]) -> List[Dict[str, str]]:
    return worlds.detect_world_drift(world_id, observed)
