"""Identity validator: frame-embedding drift against the identity matrix.
Face-landmark identity checking is a documented upgrade behind the same
signature (returns similarity in [-1, 1])."""
from __future__ import annotations

from typing import Optional

import torch

from ..memory.global_identity_matrix import GlobalIdentityMatrix
from ..memory.temporal_kv_cache import compress_frames_to_embedding


def identity_similarity(matrix: GlobalIdentityMatrix, profile_id: str,
                        frames: torch.Tensor) -> Optional[float]:
    if profile_id not in matrix.embeddings:
        return None
    candidate = compress_frames_to_embedding(
        frames, dim=matrix.embed_dim)
    return matrix.compare_identity_drift(profile_id, candidate)
