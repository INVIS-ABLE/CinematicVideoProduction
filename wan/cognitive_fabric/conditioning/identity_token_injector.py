"""Inject identity tokens from the Global Identity Matrix into a
conditioning pack (Phase 1: tensor assembly + prompt clauses; Phase 4:
cross-attention context injection via fabric_hooks.build_context_injection)."""
from __future__ import annotations

from typing import List

from ..fabric_types import CognitiveConditioningPack
from ..memory.global_identity_matrix import GlobalIdentityMatrix


def inject_identity_tokens(pack: CognitiveConditioningPack,
                           matrix: GlobalIdentityMatrix,
                           character_ids: List[str],
                           object_ids: List[str]) -> CognitiveConditioningPack:
    tokens = matrix.get_identity_tokens_for_shot(character_ids, object_ids)
    pack.identity_tokens = tokens
    pack.prompt_clauses.extend(matrix.identity_prompt_clauses(character_ids))
    pack.validate()
    return pack
