"""Motion validator: jitter/stutter/static detection from frame deltas.
Optical-flow-based sliding/warp detection is a documented upgrade point."""
from __future__ import annotations

from typing import Dict

import torch


def measure(frames: torch.Tensor) -> Dict[str, float]:
    rgb = (frames.clamp(-1, 1) + 1) / 2
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    f = luma.shape[0]
    if f < 3:
        return {"motion_mean": 0.0, "motion_jitter": 0.0, "motion_static": 1.0}
    deltas = luma.diff(dim=0).abs().mean(dim=(1, 2))  # [F-1]
    return {
        "motion_mean": float(deltas.mean()),
        "motion_jitter": float(deltas.diff().abs().mean()),
        "motion_static": float((deltas < 1e-4).float().mean()),
    }
