"""Multimodal reference token stack (spec §6).

Converts user references (images, videos, audio, character sheets, previous
frames…) into structured conditioning tokens with priority-based pruning.
Never silently ignores a reference: `compile_tokens()` returns a manifest of
what was used and what was pruned, and both are logged.

Encoders are pluggable. Phase 1 ships `DeterministicReferenceEncoder`
(content-hash-seeded embeddings — stable across runs, zero model weights,
CPU-fast) so the whole pipeline runs today; stronger frozen encoders
register through `register_encoder()` without changing any caller.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..fabric_logging import get_fabric_logger
from ..fabric_types import CognitiveConditioningPack, ReferenceToken

logger = get_fabric_logger("reference_stack")

Encoder = Callable[[str, str], torch.Tensor]  # (source_path, role) -> [D]


class DeterministicReferenceEncoder:
    """Content-addressed pseudo-embedding: same file bytes → same vector.
    Provides a real, testable identity/similarity signal (identical refs
    match exactly; different refs are near-orthogonal in high dim)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, source_path: str, role: str) -> torch.Tensor:
        hasher = hashlib.sha256()
        hasher.update(role.encode("utf-8"))
        p = Path(source_path) if source_path else None
        if p is not None and p.is_file():
            with open(p, "rb") as fh:
                # first + last 64 KiB is enough to fingerprint media files
                hasher.update(fh.read(65536))
                try:
                    fh.seek(-65536, 2)
                    hasher.update(fh.read(65536))
                except OSError:
                    pass
        else:
            hasher.update((source_path or "").encode("utf-8"))
        seed = int.from_bytes(hasher.digest()[:8], "little")
        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(self.dim, generator=gen)
        return v / v.norm().clamp_min(1e-8)


class ReferenceTokenStack:
    def __init__(self, token_limit: int = 50,
                 encoder: Optional[Encoder] = None, dim: int = 256):
        self.token_limit = token_limit
        self.dim = dim
        self._encoders: Dict[str, Encoder] = {}
        self._default_encoder: Encoder = encoder or DeterministicReferenceEncoder(dim)
        self._tokens: List[ReferenceToken] = []
        self._counter = 0

    # ---- encoder plugin point --------------------------------------------------

    def register_encoder(self, role: str, encoder: Encoder) -> None:
        self._encoders[role] = encoder

    def _encode(self, source_path: Optional[str], role: str,
                embedding: Optional[torch.Tensor]) -> torch.Tensor:
        if embedding is not None:
            return embedding
        encoder = self._encoders.get(role, self._default_encoder)
        return encoder(source_path or "", role)

    # ---- add_* API (spec-required names) -----------------------------------------

    def _add(self, role: str, source_path: Optional[str], *,
             weight: float = 1.0, priority: int = 5,
             scene_scope: Optional[str] = None,
             shot_scope: Optional[str] = None,
             character_scope: Optional[str] = None,
             object_scope: Optional[str] = None,
             temporal_validity: Optional[str] = None,
             persist: bool = False,
             embedding: Optional[torch.Tensor] = None) -> ReferenceToken:
        self._counter += 1
        token = ReferenceToken(
            id=f"tok_{self._counter:04d}", role=role, source_path=source_path,
            embedding=self._encode(source_path, role, embedding),
            weight=weight, priority=priority, scene_scope=scene_scope,
            shot_scope=shot_scope, character_scope=character_scope,
            object_scope=object_scope, temporal_validity=temporal_validity,
            persist=persist)
        token.validate()
        self._tokens.append(token)
        return token

    def add_image_reference(self, path: str, **kw: Any) -> ReferenceToken:
        return self._add("image", path, **kw)

    def add_video_reference(self, path: str, **kw: Any) -> ReferenceToken:
        return self._add("video", path, **kw)

    def add_audio_reference(self, path: str, **kw: Any) -> ReferenceToken:
        return self._add("audio", path, **kw)

    def add_character_reference(self, path: str, character_id: str,
                                **kw: Any) -> ReferenceToken:
        kw.setdefault("priority", 9)  # identity refs are nearly never pruned
        kw.setdefault("persist", True)
        return self._add("character_sheet", path,
                         character_scope=character_id, **kw)

    def add_object_reference(self, path: str, object_id: str,
                             **kw: Any) -> ReferenceToken:
        kw.setdefault("priority", 8)
        kw.setdefault("persist", True)
        return self._add("object_sheet", path, object_scope=object_id, **kw)

    def add_world_reference(self, path: str, **kw: Any) -> ReferenceToken:
        kw.setdefault("priority", 7)
        return self._add("world", path, **kw)

    def add_style_reference(self, path: str, **kw: Any) -> ReferenceToken:
        return self._add("style", path, **kw)

    def add_previous_frame_reference(self, path: str, **kw: Any) -> ReferenceToken:
        kw.setdefault("priority", 10)  # continuity anchor: never pruned
        return self._add("previous_frame", path, **kw)

    # ---- compile / prune ------------------------------------------------------------

    def prune_to_limit(self) -> List[ReferenceToken]:
        """Drop lowest-priority tokens above the limit. Returns pruned list."""
        if len(self._tokens) <= self.token_limit:
            return []
        ordered = sorted(self._tokens,
                         key=lambda t: (t.priority, t.weight), reverse=True)
        keep, pruned = ordered[:self.token_limit], ordered[self.token_limit:]
        keep_ids = {t.id for t in keep}
        self._tokens = [t for t in self._tokens if t.id in keep_ids]
        for t in pruned:
            logger.warning("pruned reference %s (%s, priority %d) — over "
                           "token limit %d", t.id, t.role, t.priority,
                           self.token_limit)
        return pruned

    def compile_tokens(self, scene_id: Optional[str] = None,
                       shot_id: Optional[str] = None
                       ) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
        """→ ([n, D] weighted token matrix or None, manifest of used refs)."""
        pruned = self.prune_to_limit()
        active = [t for t in self._tokens
                  if (t.scene_scope in (None, scene_id)) and
                     (t.shot_scope in (None, shot_id))]
        manifest = [{
            "id": t.id, "role": t.role, "source": t.source_path,
            "weight": t.weight, "priority": t.priority, "used": True,
        } for t in active]
        manifest += [{
            "id": t.id, "role": t.role, "source": t.source_path,
            "weight": t.weight, "priority": t.priority, "used": False,
            "reason": "pruned_over_limit",
        } for t in pruned]
        for entry in manifest:
            logger.info("reference %(id)s role=%(role)s used=%(used)s", entry)
        if not active:
            return None, manifest
        rows = [t.embedding.flatten()[:self.dim] * t.weight for t in active]
        return torch.stack(rows), manifest

    def to_conditioning_pack(self, scene_id: Optional[str] = None,
                             shot_id: Optional[str] = None,
                             base: Optional[CognitiveConditioningPack] = None
                             ) -> CognitiveConditioningPack:
        tokens, manifest = self.compile_tokens(scene_id, shot_id)
        pack = base or CognitiveConditioningPack()
        pack.reference_tokens = tokens
        pack.reference_manifest = manifest
        pack.validate()
        return pack

    def __len__(self) -> int:
        return len(self._tokens)
