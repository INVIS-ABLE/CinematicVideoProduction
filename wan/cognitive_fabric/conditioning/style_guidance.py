"""Style guidance: applies the locked style bible to every shot prompt."""
from __future__ import annotations

from ..fabric_types import CognitiveConditioningPack
from ..memory.style_memory import StyleMemory


def inject_style_guidance(pack: CognitiveConditioningPack,
                          style: StyleMemory) -> CognitiveConditioningPack:
    pack.prompt_clauses.extend(style.prompt_clauses())
    if style.style.get("negative"):
        pack.negative_clauses.append(style.style["negative"])
    return pack
