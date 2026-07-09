"""Per-scene working state shared between brains during generation of one
scene (camera direction, motion vector, active entities, lighting lock)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class SceneStateCache:
    def __init__(self):
        self._scenes: Dict[str, Dict[str, Any]] = {}

    def state(self, scene_id: str) -> Dict[str, Any]:
        return self._scenes.setdefault(scene_id, {
            "active_characters": [], "active_objects": [],
            "camera_direction": None, "motion_direction": None,
            "lighting_lock": None, "last_chunk_id": None,
        })

    def update(self, scene_id: str, **fields: Any) -> Dict[str, Any]:
        state = self.state(scene_id)
        state.update(fields)
        return state

    def clear(self, scene_id: Optional[str] = None) -> None:
        if scene_id is None:
            self._scenes.clear()
        else:
            self._scenes.pop(scene_id, None)
