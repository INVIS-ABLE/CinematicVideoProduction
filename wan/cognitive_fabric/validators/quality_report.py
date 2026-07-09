"""Aggregate validator entry: runs every available validator over a chunk's
frames and merges the results into a single QualityReport."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..brains.quality_brain import QualityBrain
from ..fabric_types import QualityReport
from . import flicker_validator, motion_validator


def build_quality_report(chunk_id: str, shot_id: str, scene_id: str,
                         frames: Optional[torch.Tensor],
                         quality_brain: Optional[QualityBrain] = None,
                         seam_similarity: Optional[float] = None,
                         identity_similarity: Optional[float] = None
                         ) -> QualityReport:
    brain = quality_brain or QualityBrain()
    report = brain.score_chunk(chunk_id, shot_id, scene_id, frames,
                               seam_similarity=seam_similarity,
                               identity_similarity=identity_similarity)
    if frames is not None:
        report.measured.update(flicker_validator.measure(frames))
        report.measured.update(motion_validator.measure(frames))
    return report
