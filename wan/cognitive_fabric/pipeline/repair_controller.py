"""Repair controller: applies a RepairBrain decision to a chunk spec and its
conditioning before the next attempt."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..fabric_types import ChunkSpec


def apply_repair_action(action: str, chunk: ChunkSpec,
                        conditioning: Dict[str, Any],
                        last_video: Optional[torch.Tensor]
                        ) -> Tuple[ChunkSpec, Dict[str, Any]]:
    chunk.attempt += 1
    if action == "adjust_seed":
        chunk.seed = (chunk.seed * 1103515245 + 12345) % (2 ** 31)
    elif action == "strengthen_identity_tokens":
        conditioning["prompt"] = (
            conditioning["prompt"] +
            ". strict character identity lock: the exact same face, hair, "
            "wardrobe and body as established")
    elif action == "strengthen_world_tokens":
        conditioning["prompt"] = (
            conditioning["prompt"] +
            ". strict world lock: identical location, layout, weather and "
            "time of day as established")
    elif action == "reduce_motion_complexity":
        conditioning["prompt"] = (
            conditioning["prompt"] + ". calm, simple, slow motion only")
    elif action == "reduce_camera_complexity":
        conditioning["prompt"] = (
            conditioning["prompt"] + ". static locked-off camera")
    elif action == "use_previous_frame_conditioning":
        pass  # first_frame_path already set by orchestrator when available
    elif action == "stabilise":
        conditioning["negative_prompt"] = (
            conditioning["negative_prompt"] + ", flicker, strobing, jitter")
    elif action == "detail_pass":
        conditioning["prompt"] = (
            conditioning["prompt"] +
            ". sharp fine detail, crisp focus, high micro-contrast")
    elif action in ("regenerate", "split_shot", "flag_for_user"):
        chunk.seed += 1  # split_shot/flag escalate in a later phase
    return chunk, conditioning
