"""Inject object continuity tokens/clauses into a conditioning pack."""
from __future__ import annotations

from typing import List

from ..fabric_types import CognitiveConditioningPack
from ..memory.object_memory import ObjectMemory


def inject_object_tokens(pack: CognitiveConditioningPack,
                         objects: ObjectMemory,
                         object_ids: List[str]) -> CognitiveConditioningPack:
    tokens = objects.matrix.get_identity_tokens_for_shot([], object_ids)
    pack.object_tokens = tokens
    pack.prompt_clauses.extend(objects.prompt_clauses(object_ids))
    pack.validate()
    return pack
