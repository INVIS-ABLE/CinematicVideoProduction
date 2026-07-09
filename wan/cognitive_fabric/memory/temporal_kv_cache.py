"""Long-horizon temporal memory (spec §10).

Wan must not reset memory at every 5-second clip. Per chunk the cache keeps:
terminal frames (pixel-space, for the TI2V first-frame clamp), compressed
visual embeddings, and the active identity/world token snapshots — with
temporal decay and pruning so a 60-minute film never overflows.

Phase 1 stores frames + embeddings + token snapshots (all real, tested).
Optional raw attention K/V capture is a Phase-4 hook: `capture()` accepts a
`kv_snapshot` argument today so the interface is stable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ChunkMemory:
    chunk_id: str
    scene_id: str
    terminal_frames: Optional[torch.Tensor] = None      # [3, n, H, W]
    visual_embedding: Optional[torch.Tensor] = None     # [D]
    identity_tokens: Optional[torch.Tensor] = None      # [n_id, D]
    world_tokens: Optional[torch.Tensor] = None         # [n_world, D]
    kv_snapshot: Optional[Dict[str, torch.Tensor]] = None  # Phase 4
    motion_direction: Optional[str] = None
    weight: float = 1.0
    captured_at: float = field(default_factory=time.time)


def compress_frames_to_embedding(frames: torch.Tensor,
                                 dim: int = 256) -> torch.Tensor:
    """Cheap deterministic visual embedding: per-channel spatial statistics
    pooled over frames, projected to `dim` by a fixed seeded projection.
    Good enough for drift detection between adjacent chunks; replaced by a
    learned encoder later behind the same signature."""
    if frames.dim() != 4:
        raise ValueError(f"expected [C, F, H, W], got {tuple(frames.shape)}")
    c, f, h, w = frames.shape
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        frames.reshape(c * f, 1, h, w), (8, 8)).reshape(c, f, 64).mean(dim=1)
    flat = pooled.flatten().float()  # [c*64]
    gen = torch.Generator().manual_seed(0xFAB)
    projection = torch.randn(flat.numel(), dim, generator=gen)
    return (flat @ projection) / flat.numel() ** 0.5


class TemporalKVCache:
    def __init__(self, max_cache_chunks: int = 12, decay_rate: float = 0.82,
                 scene_boundary_decay: float = 0.35,
                 preserve_identity_tokens: bool = True,
                 preserve_world_tokens: bool = True):
        self.max_cache_chunks = max_cache_chunks
        self.decay_rate = decay_rate
        self.scene_boundary_decay = scene_boundary_decay
        self.preserve_identity_tokens = preserve_identity_tokens
        self.preserve_world_tokens = preserve_world_tokens
        self._entries: List[ChunkMemory] = []

    def __len__(self) -> int:
        return len(self._entries)

    # ---- capture / retrieve -------------------------------------------------

    def capture(self, chunk_id: str, scene_id: str,
                terminal_frames: Optional[torch.Tensor] = None,
                identity_tokens: Optional[torch.Tensor] = None,
                world_tokens: Optional[torch.Tensor] = None,
                kv_snapshot: Optional[Dict[str, torch.Tensor]] = None,
                motion_direction: Optional[str] = None) -> ChunkMemory:
        visual = (compress_frames_to_embedding(terminal_frames)
                  if terminal_frames is not None else None)
        entry = ChunkMemory(
            chunk_id=chunk_id, scene_id=scene_id,
            terminal_frames=(terminal_frames.detach().clone()
                             if terminal_frames is not None else None),
            visual_embedding=visual,
            identity_tokens=(identity_tokens.detach().clone()
                             if identity_tokens is not None else None),
            world_tokens=(world_tokens.detach().clone()
                          if world_tokens is not None else None),
            kv_snapshot=kv_snapshot,
            motion_direction=motion_direction,
        )
        self._entries.append(entry)
        self.prune()
        return entry

    def retrieve(self, scene_id: Optional[str] = None,
                 last_n: int = 1) -> List[ChunkMemory]:
        pool = [e for e in self._entries
                if scene_id is None or e.scene_id == scene_id]
        return pool[-last_n:]

    def latest(self) -> Optional[ChunkMemory]:
        return self._entries[-1] if self._entries else None

    # ---- decay / prune --------------------------------------------------------

    def decay(self, scene_boundary: bool = False) -> None:
        rate = self.scene_boundary_decay if scene_boundary else self.decay_rate
        for entry in self._entries:
            entry.weight *= rate
            if entry.weight < 0.05:
                if not self.preserve_identity_tokens:
                    entry.identity_tokens = None
                if not self.preserve_world_tokens:
                    entry.world_tokens = None
                entry.terminal_frames = None  # heaviest payload drops first
                entry.kv_snapshot = None

    def prune(self) -> None:
        if len(self._entries) > self.max_cache_chunks:
            self._entries = self._entries[-self.max_cache_chunks:]

    def clear_scene(self, scene_id: str) -> None:
        for entry in self._entries:
            if entry.scene_id == scene_id:
                entry.terminal_frames = None
                entry.kv_snapshot = None
                entry.weight = 0.0

    # ---- continuity signal ------------------------------------------------------

    def continuity_similarity(self, frames: torch.Tensor) -> Optional[float]:
        """Cosine similarity between a new chunk's opening frames and the
        cached previous terminal embedding — a cheap seam-quality signal."""
        latest = self.latest()
        if latest is None or latest.visual_embedding is None:
            return None
        candidate = compress_frames_to_embedding(frames)
        a, b = latest.visual_embedding, candidate
        return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))

    # ---- checkpoint ------------------------------------------------------------

    def save_checkpoint(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": {
                "max_cache_chunks": self.max_cache_chunks,
                "decay_rate": self.decay_rate,
                "scene_boundary_decay": self.scene_boundary_decay,
                "preserve_identity_tokens": self.preserve_identity_tokens,
                "preserve_world_tokens": self.preserve_world_tokens,
            },
            "entries": [{
                "chunk_id": e.chunk_id,
                "scene_id": e.scene_id,
                "terminal_frames": e.terminal_frames,
                "visual_embedding": e.visual_embedding,
                "identity_tokens": e.identity_tokens,
                "world_tokens": e.world_tokens,
                "motion_direction": e.motion_direction,
                "weight": e.weight,
                "captured_at": e.captured_at,
            } for e in self._entries],
        }, p)
        return p

    @classmethod
    def load_checkpoint(cls, path: str) -> "TemporalKVCache":
        payload = torch.load(path, weights_only=True)
        cache = cls(**payload["config"])
        for raw in payload["entries"]:
            cache._entries.append(ChunkMemory(**raw))
        return cache
