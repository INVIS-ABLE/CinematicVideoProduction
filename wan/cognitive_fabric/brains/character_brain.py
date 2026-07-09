"""Character Brain: registers storyboard characters into the identity
matrix and produces per-shot identity conditioning."""
from __future__ import annotations

from typing import Any, Dict, List

from ..memory.character_memory import CharacterMemory


class CharacterBrain:
    name = "character"
    kind = "brain"

    def __init__(self, memory: CharacterMemory):
        self.memory = memory

    def health_check(self) -> Dict[str, Any]:
        return self.memory.health_check()

    def register_characters(self, storyboard: Dict[str, Any]) -> List[str]:
        registered = []
        for char in storyboard.get("characters", []):
            self.memory.matrix.register_character(
                char["character_id"], char.get("name", char["character_id"]),
                face_description=char.get("face_description", ""),
                wardrobe_description=char.get("wardrobe_description", ""),
                colour_palette=char.get("colour_palette", ""),
                continuity_rules=char.get("continuity_rules", []))
            registered.append(char["character_id"])
        return registered
