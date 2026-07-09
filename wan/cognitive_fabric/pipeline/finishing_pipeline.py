"""Finishing pipeline: colour finish, grain, letterbox, export prep — the
last pass before delivery. Wraps HierarchicalScaler primitives with named
looks; OCIO/ACES integration is a documented upgrade point."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def finish_video_file(input_path: str, output_path: str, *,
                      look: str = "neutral",
                      target_size: Optional[Tuple[int, int]] = None,
                      fps: Optional[float] = None,
                      batch: int = 16,
                      grain: float = 0.0,
                      sharpen: float = 0.0) -> str:
    """Streaming finishing/upscale pass over an assembled video file.

    Reads `batch` frames at a time (a 60-minute master never has to fit in
    RAM), optionally upscales through the tiled hierarchical scaler to
    `target_size` (width, height), applies the named look, and writes an
    H.264 file. Requires imageio + ffmpeg; callers treat failure as
    "finishing skipped", never as a fatal pipeline error.
    """
    import imageio.v2 as iio
    import numpy as np

    reader = iio.get_reader(input_path)
    meta = reader.get_meta_data()
    fps = fps or meta.get("fps", 24)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(out), fps=fps, codec="libx264",
                            pixelformat="yuv420p")

    pipeline = FinishingPipeline()

    def flush(frames_np) -> None:
        if not frames_np:
            return
        stack = torch.from_numpy(np.stack(frames_np)).float()  # [F, H, W, 3]
        video = stack.permute(3, 0, 1, 2) / 127.5 - 1.0        # [3, F, H, W]
        if target_size is not None:
            video = pipeline.scaler._enhance_video(video, target_size)
        video = pipeline.finish(video, look, grain=grain, sharpen=sharpen)
        frames_out = ((video.clamp(-1, 1) + 1) * 127.5).round().byte()
        frames_out = frames_out.permute(1, 2, 3, 0).cpu().numpy()
        for frame in frames_out:
            writer.append_data(frame)

    buffer = []
    try:
        for frame in reader:
            buffer.append(frame[..., :3])  # drop alpha if present
            if len(buffer) >= batch:
                flush(buffer)
                buffer = []
        flush(buffer)
    finally:
        writer.close()
        reader.close()
    return str(out)
