"""Frame interpolation (24 → 48/60 fps). Phase 1 backend is linear frame
blending — real, seam-free, honest about its softness. RIFE registers via
`register_backend()` without changing callers."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

BlendFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def _linear_blend(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a * (1 - t) + b * t


class FrameInterpolator:
    def __init__(self):
        self._backends: Dict[str, BlendFn] = {"linear": _linear_blend}
        self.backend = "linear"

    def register_backend(self, name: str, fn: BlendFn) -> None:
        self._backends[name] = fn

    def interpolate(self, video: torch.Tensor, factor: int = 2) -> torch.Tensor:
        """[3, F, H, W] → [3, (F-1)*factor+1, H, W]."""
        if factor < 2 or video.shape[1] < 2:
            return video
        blend = self._backends[self.backend]
        frames = []
        for i in range(video.shape[1] - 1):
            frames.append(video[:, i])
            for k in range(1, factor):
                frames.append(blend(video[:, i], video[:, i + 1], k / factor))
        frames.append(video[:, -1])
        return torch.stack(frames, dim=1)
