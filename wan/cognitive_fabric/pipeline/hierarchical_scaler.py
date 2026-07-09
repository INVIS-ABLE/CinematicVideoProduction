"""Hierarchical scaler (spec §16): progressive 720p → 1080p → 1440p → 4K.

Phase 1 backend is real tiled bicubic upsampling with feathered overlap
blending and temporal smoothing — modest quality, honest, zero weights,
works today. ESRGAN/VEnhancer/RIFE register through `register_backend()`
behind the same tile-safe interface.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

EnhanceFn = Callable[[torch.Tensor], torch.Tensor]  # [C,h,w] -> [C,h*s,w*s]

STAGES: Dict[str, Tuple[int, int]] = {
    "hd": (1920, 1080),
    "qhd": (2560, 1440),
    "uhd": (3840, 2160),
}


def _bicubic_backend(scale: int) -> EnhanceFn:
    def enhance(tile: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tile.unsqueeze(0), scale_factor=scale,
                             mode="bicubic", align_corners=False
                             ).squeeze(0).clamp(-1, 1)
    return enhance


def _feather_mask(h: int, w: int, overlap: int,
                  device: torch.device) -> torch.Tensor:
    mask = torch.ones(h, w, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap + 2, device=device)[1:-1]
        mask[:overlap, :] *= ramp.view(-1, 1)
        mask[-overlap:, :] *= ramp.flip(0).view(-1, 1)
        mask[:, :overlap] *= ramp.view(1, -1)
        mask[:, -overlap:] *= ramp.flip(0).view(1, -1)
    return mask


def tiled_enhance(frame: torch.Tensor, enhance: EnhanceFn, scale: int,
                  tile: int = 512, overlap: int = 64) -> torch.Tensor:
    """Apply `enhance` per overlapping tile with feathered blending.
    frame: [C, H, W] → [C, H*scale, W*scale], no visible seams."""
    c, h, w = frame.shape
    if h <= tile and w <= tile:
        return enhance(frame)
    out = torch.zeros(c, h * scale, w * scale, device=frame.device)
    weight = torch.zeros(1, h * scale, w * scale, device=frame.device)
    step = tile - overlap
    ys = list(range(0, max(h - overlap, 1), step))
    xs = list(range(0, max(w - overlap, 1), step))
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            y0a, x0a = max(0, y1 - tile), max(0, x1 - tile)
            patch = enhance(frame[:, y0a:y1, x0a:x1])
            mask = _feather_mask(patch.shape[1], patch.shape[2],
                                 overlap * scale, frame.device)
            out[:, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += patch * mask
            weight[:, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += mask
    return out / weight.clamp_min(1e-6)


class HierarchicalScaler:
    def __init__(self, tile: int = 512, overlap: int = 64):
        self.tile = tile
        self.overlap = overlap
        self._backends: Dict[str, EnhanceFn] = {}

    def register_backend(self, name: str, fn: EnhanceFn) -> None:
        self._backends[name] = fn

    def _enhance_video(self, video: torch.Tensor,
                       target: Tuple[int, int],
                       backend: Optional[str] = None) -> torch.Tensor:
        """video [3, F, H, W] → [3, F, th, tw] (integer scale + resize fit)."""
        tw, th = target
        c, f, h, w = video.shape
        scale = max(1, min(tw // w, th // h))
        fn = self._backends.get(backend) if backend else None
        fn = fn or _bicubic_backend(scale)
        frames = [tiled_enhance(video[:, i], fn, scale,
                                self.tile, self.overlap) for i in range(f)]
        out = torch.stack(frames, dim=1)
        if out.shape[-2:] != (th, tw):
            out = F.interpolate(out.permute(1, 0, 2, 3), size=(th, tw),
                                mode="bicubic", align_corners=False
                                ).permute(1, 0, 2, 3).clamp(-1, 1)
        return out

    def upscale_to_1080p(self, video, backend=None):
        return self._enhance_video(video, STAGES["hd"], backend)

    def upscale_to_1440p(self, video, backend=None):
        return self._enhance_video(video, STAGES["qhd"], backend)

    def upscale_to_4k(self, video, backend=None):
        return self._enhance_video(video, STAGES["uhd"], backend)

    def denoise_temporal(self, video: torch.Tensor,
                         strength: float = 0.25) -> torch.Tensor:
        """Neighbour-frame blend — reduces shimmer without ghosting at low
        strength. RIFE-grade interpolation registers as a backend later."""
        if video.shape[1] < 3 or strength <= 0:
            return video
        out = video.clone()
        out[:, 1:-1] = (video[:, 1:-1] * (1 - strength) +
                        (video[:, :-2] + video[:, 2:]) * (strength / 2))
        return out

    def sharpen_temporal(self, video: torch.Tensor,
                         amount: float = 0.3) -> torch.Tensor:
        blurred = F.avg_pool2d(
            video.permute(1, 0, 2, 3), 3, stride=1, padding=1)
        return (video + amount *
                (video - blurred.permute(1, 0, 2, 3))).clamp(-1, 1)

    def apply_colour_finish(self, video: torch.Tensor, *,
                            lift: float = 0.0, gamma: float = 1.0,
                            gain: float = 1.0,
                            saturation: float = 1.0) -> torch.Tensor:
        rgb = (video.clamp(-1, 1) + 1) / 2
        rgb = (rgb ** (1.0 / max(gamma, 1e-3))) * gain + lift
        if saturation != 1.0:
            luma = rgb.mean(dim=0, keepdim=True)
            rgb = luma + (rgb - luma) * saturation
        return (rgb.clamp(0, 1) * 2 - 1)
