"""Multimodal reference compiler: user uploads → reference token stack, via
the reference brain's role routing."""
from __future__ import annotations

from typing import Any, Dict, List

from ..brains.reference_brain import ReferenceBrain
from ..conditioning.reference_token_stack import ReferenceTokenStack
from ..memory.reference_memory import ReferenceMemory


def compile_references(uploads: List[Dict[str, str]],
                       token_limit: int = 50) -> Dict[str, Any]:
    """uploads: [{"path": ..., "role": "character|image|video|audio|..."}]"""
    memory = ReferenceMemory()
    stack = ReferenceTokenStack(token_limit=token_limit)
    brain = ReferenceBrain(memory, stack)
    for upload in uploads:
        brain.ingest(upload["path"], upload.get("role", "auto"))
    tokens, manifest = stack.compile_tokens()
    return {"memory": memory, "stack": stack, "tokens": tokens,
            "manifest": manifest}
