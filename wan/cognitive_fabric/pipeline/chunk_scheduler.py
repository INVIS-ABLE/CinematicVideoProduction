"""Chunk scheduler: shots → Wan-sized generation chunks.

Respects the engine invariants documented in the tensor map:
frame_num ≡ 1 (mod 4), width/height multiples of the task grid.
Long shots split into chunk_seconds pieces with overlap_frames recorded for
stitch-time blending; every chunk after the first carries a first-frame
dependency on its predecessor (the TI2V latent-clamp continuity mechanism).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from ..fabric_types import ChunkSpec


def snap_frame_num(frames: int) -> int:
    """Snap to the nearest valid 4n+1 count, minimum 5."""
    n = max(1, round((frames - 1) / 4))
    return 4 * n + 1


def _seed_for(shot_id: str, chunk_index: int, base_seed: int,
              strategy: str) -> int:
    if strategy == "fixed":
        return base_seed
    digest = hashlib.sha256(f"{base_seed}:{shot_id}".encode()).digest()
    shot_seed = int.from_bytes(digest[:4], "little")
    if strategy == "per_shot":
        return shot_seed
    # per_chunk
    return shot_seed + chunk_index


def plan_chunks(shot: Dict[str, Any], *, fps: int, chunk_seconds: float,
                overlap_frames: int, width: int, height: int,
                base_seed: int = 42) -> List[ChunkSpec]:
    duration = float(shot["duration_seconds"])
    total_frames = snap_frame_num(int(round(duration * fps)))
    chunk_frames = snap_frame_num(int(round(chunk_seconds * fps)))

    chunks: List[ChunkSpec] = []
    remaining = total_frames
    index = 0
    while remaining > 0:
        frames = min(chunk_frames, snap_frame_num(remaining))
        # avoid a tiny trailing chunk: fold remainders < 1s into the previous
        if remaining - frames > 0 and remaining - frames < fps:
            frames = snap_frame_num(remaining)
        chunk_id = f"{shot['shot_id']}_c{index:02d}"
        chunks.append(ChunkSpec(
            chunk_id=chunk_id,
            shot_id=shot["shot_id"],
            scene_id=shot.get("scene_id", "scene_001"),
            index=index,
            frame_num=frames,
            fps=fps,
            width=width,
            height=height,
            seed=_seed_for(shot["shot_id"], index, base_seed,
                           shot.get("seed_strategy", "per_shot")),
            prompt=shot["prompt"],
            negative_prompt=shot.get("negative_prompt", ""),
            first_frame_path=None,  # filled at run time from predecessor
            overlap_frames=overlap_frames if index > 0 else 0,
        ))
        remaining -= frames
        index += 1
    for chunk in chunks:
        chunk.validate()
    return chunks


def plan_storyboard_chunks(storyboard: Dict[str, Any], *, fps: int,
                           chunk_seconds: float, overlap_frames: int,
                           width: int, height: int,
                           base_seed: int = 42) -> List[ChunkSpec]:
    all_chunks: List[ChunkSpec] = []
    for shot in storyboard.get("shots", []):
        all_chunks.extend(plan_chunks(
            shot, fps=fps, chunk_seconds=chunk_seconds,
            overlap_frames=overlap_frames, width=width, height=height,
            base_seed=base_seed))
    return all_chunks
