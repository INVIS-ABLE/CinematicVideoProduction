"""Finishing pipeline: colour finish, grain, letterbox, export prep — the
last pass before delivery. Wraps HierarchicalScaler primitives with named
looks; OCIO/ACES integration is a documented upgrade point."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .hierarchical_scaler import HierarchicalScaler

LOOKS: Dict[str, Dict[str, float]] = {
    "neutral": {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0},
    "cinematic_teal_orange": {"lift": -0.02, "gamma": 1.05, "gain": 1.03,
                              "saturation": 1.12},
    "documentary": {"lift": 0.0, "gamma": 0.98, "gain": 1.0, "saturation": 0.95},
    "noir": {"lift": -0.05, "gamma": 1.1, "gain": 1.05, "saturation": 0.4},
    "anime_vivid": {"lift": 0.01, "gamma": 0.95, "gain": 1.05, "saturation": 1.3},
}


class FinishingPipeline:
    def __init__(self, scaler: Optional[HierarchicalScaler] = None):
        self.scaler = scaler or HierarchicalScaler()

    def add_grain(self, video: torch.Tensor, amount: float = 0.015,
                  seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        grain = torch.randn(video.shape, generator=gen) * amount
        return (video + grain).clamp(-1, 1)

    def letterbox(self, video: torch.Tensor,
                  target_aspect: float = 2.35) -> torch.Tensor:
        c, f, h, w = video.shape
        content_h = int(round(w / target_aspect))
        if content_h >= h:
            return video
        bar = (h - content_h) // 2
        out = torch.full_like(video, -1.0)
        out[:, :, bar:bar + content_h] = video[:, :, bar:bar + content_h]
        return out

    def finish(self, video: torch.Tensor, look: str = "neutral", *,
               grain: float = 0.0, letterbox_aspect: Optional[float] = None,
               sharpen: float = 0.0) -> torch.Tensor:
        params = LOOKS.get(look, LOOKS["neutral"])
        out = self.scaler.apply_colour_finish(video, **params)
        if sharpen > 0:
            out = self.scaler.sharpen_temporal(out, sharpen)
        if grain > 0:
            out = self.add_grain(out, grain)
        if letterbox_aspect:
            out = self.letterbox(out, letterbox_aspect)
        return out
