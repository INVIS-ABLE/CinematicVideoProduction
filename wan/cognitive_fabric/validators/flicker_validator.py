"""Flicker validator: real per-frame luminance instability metrics."""
from __future__ import annotations

from typing import Dict

import torch


def measure(frames: torch.Tensor) -> Dict[str, float]:
    """frames [3, F, H, W] in [-1, 1] → global + patchwise flicker stats."""
    rgb = (frames.clamp(-1, 1) + 1) / 2
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  # [F, H, W]
    if luma.shape[0] < 2:
        return {"flicker_global": 0.0, "flicker_patch_max": 0.0}
    global_flicker = float(luma.mean(dim=(1, 2)).diff().abs().mean())
    patches = torch.nn.functional.adaptive_avg_pool2d(
        luma.unsqueeze(1), (8, 8)).squeeze(1)          # [F, 8, 8]
    patch_flicker = patches.diff(dim=0).abs().mean(dim=0)  # [8, 8]
    return {
        "flicker_global": global_flicker,
        "flicker_patch_max": float(patch_flicker.max()),
    }
