"""Hook points where the fabric touches the real Wan pipeline.

Phase 1 hooks (live):
  * attention_backend_report() — surfaces which attention path the DiT will
    use (flash3 / flash2 / SDPA fallback added in refactor R-002/R-004).
  * PipelineHooks — pre/post chunk callbacks used by the generation
    controller; post_chunk captures terminal frames for continuity.

Phase 4 hooks (interfaces, tested, documented in roadmap):
  * conditioning-pack injection into cross-attention context.
  * physics/collision attention bias.
  * temporal KV capture inside self-attention.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

HookFn = Callable[[Dict[str, Any]], None]


def attention_backend_report() -> Dict[str, Any]:
    from wan.modules.attention import (
        FLASH_ATTN_2_AVAILABLE,
        FLASH_ATTN_3_AVAILABLE,
    )
    backend = ("flash_attn_3" if FLASH_ATTN_3_AVAILABLE else
               "flash_attn_2" if FLASH_ATTN_2_AVAILABLE else
               "torch_sdpa_fallback")
    return {
        "flash_attn_3": FLASH_ATTN_3_AVAILABLE,
        "flash_attn_2": FLASH_ATTN_2_AVAILABLE,
        "cuda": torch.cuda.is_available(),
        "selected_backend": backend,
        "note": ("DiT routes through wan.modules.attention.attention() "
                 "(refactor R-002); SDPA fallback preserves dtype (R-004)."),
    }


class PipelineHooks:
    """Callback registry the generation controller drives around every
    Wan generate() call."""

    def __init__(self):
        self._pre_chunk: List[HookFn] = []
        self._post_chunk: List[HookFn] = []

    def on_pre_chunk(self, fn: HookFn) -> None:
        self._pre_chunk.append(fn)

    def on_post_chunk(self, fn: HookFn) -> None:
        self._post_chunk.append(fn)

    def run_pre_chunk(self, ctx: Dict[str, Any]) -> None:
        for fn in self._pre_chunk:
            fn(ctx)

    def run_post_chunk(self, ctx: Dict[str, Any]) -> None:
        for fn in self._post_chunk:
            fn(ctx)


def extract_terminal_frames(video: torch.Tensor,
                            count: int = 8) -> torch.Tensor:
    """Keep the last `count` frames of a decoded clip [3, F, H, W] for
    chunk-to-chunk continuity (fed back through TI2V's first-frame clamp)."""
    if video.dim() != 4:
        raise ValueError(f"expected [C, F, H, W], got {tuple(video.shape)}")
    count = max(1, min(count, video.shape[1]))
    return video[:, -count:].clone()


def terminal_frame_to_image(video: torch.Tensor):
    """Last frame of [3, F, H, W] in [-1, 1] → PIL.Image for WanTI2V.i2v(img=…)."""
    from PIL import Image
    frame = video[:, -1]  # [3, H, W]
    frame = ((frame.clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())


# ---------------------------------------------------------------------------
# Phase 4 planned hooks — tensor-shape-safe signatures, not yet wired into
# WanAttentionBlock. Tests assert the contracts so wiring cannot drift.
# ---------------------------------------------------------------------------


def build_context_injection(context: torch.Tensor,
                            extra_tokens: Optional[torch.Tensor]) -> torch.Tensor:
    """Append fabric tokens to a cross-attention context [B, L2, C].

    extra_tokens: [n, C] (already projected to text embedding dim). This is
    the Phase-4 injection route chosen in the recon (§14): context length is
    free, video-token length is not.
    """
    if extra_tokens is None:
        return context
    if extra_tokens.dim() != 2 or extra_tokens.shape[-1] != context.shape[-1]:
        raise ValueError(
            f"extra tokens [n, C] must match context channel dim "
            f"{context.shape[-1]}, got {tuple(extra_tokens.shape)}")
    b = context.shape[0]
    stacked = extra_tokens.unsqueeze(0).expand(b, -1, -1).to(
        dtype=context.dtype, device=context.device)
    return torch.cat([context, stacked], dim=1)
