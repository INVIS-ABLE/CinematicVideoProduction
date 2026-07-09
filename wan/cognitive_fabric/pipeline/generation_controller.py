"""Generation controller: the engine seam between the fabric and Wan.

Engines implement one method:
    generate_chunk(chunk, conditioning, first_frame) -> torch.Tensor [3, F, H, W]

`WanTI2VEngine` drives the real Wan 2.2 TI2V-5B pipeline (text→video for a
shot's first chunk; image→video with the predecessor's terminal frame for
every continuation chunk — Wan's own latent-clamp mechanism doing the
continuity work).

`MockWanEngine` produces deterministic seeded synthetic footage with real
motion so the entire fabric — chunking, conditioning, quality scoring,
repair, stitching, checkpoint/resume — runs and is testable on any machine
with zero model weights. It never pretends to be the real model: every
output is tagged engine="mock" in the run report.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import torch

from ..fabric_logging import get_fabric_logger
from ..fabric_types import ChunkSpec, CognitiveConditioningPack

logger = get_fabric_logger("engine")


class VideoEngine(Protocol):
    name: str

    def generate_chunk(self, chunk: ChunkSpec,
                       conditioning: Dict[str, Any],
                       first_frame: Optional[Any] = None) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# Mock engine — real pipeline exercise, honest labelling
# ---------------------------------------------------------------------------


class MockWanEngine:
    name = "mock"

    def __init__(self, downscale: int = 8):
        # generate at reduced size to keep CI fast; aspect preserved
        self.downscale = max(1, downscale)

    def generate_chunk(self, chunk: ChunkSpec,
                       conditioning: Dict[str, Any],
                       first_frame: Optional[Any] = None) -> torch.Tensor:
        # keep encoder-friendly dims (multiples of 16)
        w = max(32, (chunk.width // self.downscale) // 16 * 16)
        h = max(32, (chunk.height // self.downscale) // 16 * 16)
        f = chunk.frame_num
        digest = hashlib.sha256(chunk.prompt.encode("utf-8")).digest()
        hue = digest[0] / 255.0

        gen = torch.Generator().manual_seed(chunk.seed)
        base = torch.rand(3, 1, h, w, generator=gen) * 0.25

        ys = torch.linspace(0, 1, h).view(1, 1, h, 1)
        xs = torch.linspace(0, 1, w).view(1, 1, 1, w)
        t = torch.linspace(0, 1, f).view(1, f, 1, 1)

        # drifting diagonal gradient + moving highlight = real motion for the
        # quality metrics to measure
        phase = 2 * math.pi * (xs + ys + 0.35 * t + hue)
        r = 0.5 + 0.45 * torch.sin(phase)
        g = 0.5 + 0.45 * torch.sin(phase + 2.1)
        b = 0.5 + 0.45 * torch.sin(phase + 4.2)
        video = torch.cat([r, g, b], dim=0).expand(3, f, h, w).clone()
        cx = 0.2 + 0.6 * t
        highlight = torch.exp(-((xs - cx) ** 2 + (ys - 0.5) ** 2) / 0.02)
        video = (video * 0.8 + highlight * 0.6 + base).clamp(0, 1)

        if first_frame is not None:
            # honour the continuity contract: open on the handoff frame
            prev = _pil_to_tensor(first_frame, h, w)
            video[:, 0] = prev
            if f > 1:
                video[:, 1] = 0.5 * prev + 0.5 * video[:, 1]
        return video * 2.0 - 1.0  # [-1, 1] like the real VAE decode


def _pil_to_tensor(img: Any, h: int, w: int) -> torch.Tensor:
    import numpy as np
    img = img.convert("RGB").resize((w, h))
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Real Wan TI2V engine
# ---------------------------------------------------------------------------


class WanTI2VEngine:
    """Drives wan.WanTI2V. Requires a downloaded Wan2.2-TI2V-5B checkpoint
    and CUDA; raises a clear error otherwise (callers fall back to mock)."""

    name = "wan-ti2v-5b"

    def __init__(self, ckpt_dir: str, *, device_id: int = 0,
                 offload_model: bool = True, sample_steps: Optional[int] = None,
                 sample_solver: str = "unipc",
                 convert_model_dtype: bool = True):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "WanTI2VEngine requires CUDA. Use MockWanEngine for CPU "
                "dry-runs; the fabric selects it automatically.")
        ckpt = Path(ckpt_dir)
        if not ckpt.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

        import wan as wan_pkg
        from wan.configs import WAN_CONFIGS

        self.config = WAN_CONFIGS["ti2v-5B"]
        self.sample_steps = sample_steps or self.config.sample_steps
        self.sample_solver = sample_solver
        self.offload_model = offload_model
        logger.info("loading Wan2.2 TI2V-5B from %s", ckpt_dir)
        self.pipeline = wan_pkg.WanTI2V(
            config=self.config,
            checkpoint_dir=str(ckpt),
            device_id=device_id,
            rank=0,
            convert_model_dtype=convert_model_dtype,
        )

    def generate_chunk(self, chunk: ChunkSpec,
                       conditioning: Dict[str, Any],
                       first_frame: Optional[Any] = None) -> torch.Tensor:
        guide_scale = conditioning.get("guide_scale",
                                       self.config.sample_guide_scale)
        video = self.pipeline.generate(
            conditioning["prompt"],
            img=first_frame,  # None → t2v; PIL image → i2v latent clamp
            size=(chunk.width, chunk.height),
            max_area=chunk.width * chunk.height,
            frame_num=chunk.frame_num,
            shift=self.config.sample_shift,
            sample_solver=self.sample_solver,
            sampling_steps=self.sample_steps,
            guide_scale=guide_scale,
            n_prompt=conditioning["negative_prompt"],
            seed=chunk.seed,
            offload_model=self.offload_model,
        )
        return video  # [3, F, H, W] in [-1, 1]


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------


def save_chunk_video(video: torch.Tensor, path: str, fps: int) -> str:
    """Write [3, F, H, W] in [-1, 1] to mp4 (imageio-ffmpeg). Falls back to
    a .pt tensor when no encoder is available — the pipeline keeps running
    and the manifest records which format was written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, 3]
    try:
        import imageio.v3 as iio
        iio.imwrite(str(p.with_suffix(".mp4")), frames, fps=fps,
                    codec="libx264")  # writer defaults to yuv420p
        return str(p.with_suffix(".mp4"))
    except Exception as e:  # encoder missing — degrade, don't die
        logger.warning("mp4 encode unavailable (%s); saving tensor instead", e)
        torch.save(video, p.with_suffix(".pt"))
        return str(p.with_suffix(".pt"))


def extract_last_frame_png(video: torch.Tensor, path: str) -> Optional[str]:
    """Persist the terminal frame as PNG for the next chunk's first-frame
    conditioning. Returns None (and keeps going) if no imaging lib exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame = ((video[:, -1].clamp(-1, 1) + 1) * 127.5).round().byte()
    try:
        from PIL import Image
        Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()).save(p)
        return str(p)
    except Exception as e:
        logger.warning("could not write terminal frame png: %s", e)
        return None
