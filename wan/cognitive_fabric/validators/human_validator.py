"""Human validator — planned hook (see fable_memory/roadmap.md).

Honest status: reliable human scoring needs dedicated local models
(pose/landmark/segmentation). This module pins the interface the repair
loop already consumes; until the real detector lands it declines to score
(returns None) instead of inventing numbers.
"""
from __future__ import annotations

from typing import Optional

import torch


def score(frames: torch.Tensor) -> Optional[float]:
    """[3, F, H, W] -> score in [0, 1], or None when unavailable."""
    if frames.dim() != 4:
        raise ValueError(f"expected [C, F, H, W], got {tuple(frames.shape)}")
    return None  # not yet measurable locally — do not fake


AVAILABLE = False
