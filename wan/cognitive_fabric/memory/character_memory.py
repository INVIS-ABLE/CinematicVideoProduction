"""Character memory — domain wrapper over the Global Identity Matrix that
extracts characters from prompts/references and keeps shot history."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .global_identity_matrix import GlobalIdentityMatrix


class CharacterMemory:
    def __init__(self, matrix: GlobalIdentityMatrix):
        self.matrix = matrix
        self.shot_history: Dict[str, List[str]] = {}

    def register_from_description(self, character_id: str, name: str,
                                  description: str, **kwargs: Any) -> Dict[str, Any]:
        return self.matrix.register_character(
            character_id, name, face_description=description, **kwargs)

    def mark_appearance(self, character_id: str, scene_id: str,
                        shot_id: str) -> None:
        profile = self.matrix.characters.get(character_id)
        if profile is None:
            raise KeyError(f"unknown character: {character_id}")
        profile["last_seen_scene"] = scene_id
        profile["last_seen_shot"] = shot_id
        self.shot_history.setdefault(character_id, []).append(shot_id)

    def prompt_clauses(self, character_ids: List[str]) -> List[str]:
        return self.matrix.identity_prompt_clauses(character_ids)

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "characters": len(self.matrix.characters)}
