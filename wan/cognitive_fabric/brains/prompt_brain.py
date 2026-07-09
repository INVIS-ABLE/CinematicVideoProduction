"""Prompt Brain: assembles the final per-chunk prompt and negative prompt
from the user's idea plus fabric conditioning clauses.

Rule from the spec: preserve the user's original idea — enhancement adds
precision, it never hijacks creativity. The user prompt always leads; fabric
clauses follow, deduplicated and length-capped.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..fabric_types import CognitiveConditioningPack

# English cinematic anti-artifact negative (upstream default is a Chinese
# boilerplate in shared_config.py; we merge rather than replace so existing
# tuned behaviour is kept when the pipeline falls back to config defaults).
DEFAULT_NEGATIVE = (
    "warped face, deformed hands, extra fingers, broken anatomy, "
    "identity drift, changing clothes, floating objects, object popping, "
    "flickering, jitter, frame strobing, sliding feet, inconsistent lighting, "
    "shadows pointing the wrong way, background morphing, low detail, "
    "oversaturated, overexposed, watermark, subtitles, static image"
)

MAX_PROMPT_CHARS = 1800  # stay well inside umT5's 512-token budget


class PromptBrain:
    name = "prompt"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}

    @staticmethod
    def _dedup(clauses: List[str]) -> List[str]:
        seen, out = set(), []
        for clause in clauses:
            key = clause.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(clause.strip())
        return out

    def compile_prompt(self, user_prompt: str,
                       pack: CognitiveConditioningPack) -> str:
        parts = [user_prompt.strip().rstrip(".")]
        parts.extend(self._dedup(pack.prompt_clauses))
        text = ". ".join(parts)
        if len(text) > MAX_PROMPT_CHARS:
            # never truncate the user's idea — trim fabric clauses instead
            budget = MAX_PROMPT_CHARS - len(parts[0]) - 2
            kept: List[str] = []
            for clause in self._dedup(pack.prompt_clauses):
                if budget - len(clause) - 2 < 0:
                    break
                kept.append(clause)
                budget -= len(clause) + 2
            text = ". ".join([parts[0]] + kept)
        return text

    def compile_negative(self, user_negative: str,
                         pack: CognitiveConditioningPack) -> str:
        clauses = []
        if user_negative.strip():
            clauses.append(user_negative.strip().rstrip(","))
        clauses.append(DEFAULT_NEGATIVE)
        clauses.extend(pack.negative_clauses)
        return ", ".join(self._dedup(clauses))
