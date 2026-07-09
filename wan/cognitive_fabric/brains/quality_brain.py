"""Quality Brain: scores generated chunks.

Phase 1 measures what can honestly be measured today from pixels without
extra model weights — temporal flicker, motion coherence, sharpness,
exposure sanity, frame-to-frame identity stability via the deterministic
embedding — and reports everything else as 0.0/absent rather than faking
scores. VBench-style semantic scoring plugs into `score_chunk` later.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..fabric_types import QualityReport

_LAPLACIAN = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
                          ).view(1, 1, 3, 3)


def _luma(frames: torch.Tensor) -> torch.Tensor:
    """[3, F, H, W] in [-1, 1] → [F, H, W] luminance in [0, 1]."""
    rgb = (frames.clamp(-1, 1) + 1) / 2
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


class QualityBrain:
    name = "quality"
    kind = "brain"

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {}

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok",
                "measures": ["flicker", "motion", "sharpness", "exposure",
                             "seam_similarity"],
                "planned": ["vbench", "face_consistency", "prompt_match_clip"]}

    # ---- raw metrics ---------------------------------------------------------

    def measure(self, frames: torch.Tensor) -> Dict[str, float]:
        """frames: [3, F, H, W] in [-1, 1]."""
        if frames.dim() != 4 or frames.shape[0] != 3:
            raise ValueError(f"expected [3, F, H, W], got {tuple(frames.shape)}")
        luma = _luma(frames)  # [F, H, W]
        f = luma.shape[0]

        # flicker: second-order difference of mean luminance — measures
        # oscillation, not smooth exposure ramps (a linear brightness ramp
        # from camera motion has ~zero second difference; alternating-frame
        # strobing scores high)
        means = luma.mean(dim=(1, 2))
        flicker_raw = float(means.diff().diff().abs().mean()) if f > 2 else 0.0
        flicker_score = max(0.0, 1.0 - flicker_raw * 30)

        # motion: mean absolute pixel change — 0 motion and chaotic motion
        # both score low, smooth moderate motion scores high
        if f > 1:
            deltas = luma.diff(dim=0).abs().mean(dim=(1, 2))  # [F-1]
            motion_mag = float(deltas.mean())
            motion_smoothness = 1.0 - min(1.0, float(deltas.diff().abs().mean()
                                                     ) * 60) if f > 2 else 1.0
            motion_presence = min(1.0, motion_mag * 50)
            motion_score = 0.5 * motion_presence + 0.5 * motion_smoothness
        else:
            motion_mag, motion_score = 0.0, 0.0

        # sharpness: laplacian energy, normalised with a soft knee
        lap = F.conv2d(luma.unsqueeze(1), _LAPLACIAN.to(luma), padding=1)
        sharp_raw = float(lap.abs().mean())
        sharpness_score = min(1.0, sharp_raw * 25)

        # exposure sanity: fraction of pixels not crushed/clipped
        ok = ((luma > 0.02) & (luma < 0.98)).float().mean()
        exposure_score = float(ok)

        return {
            "flicker_raw": flicker_raw, "flicker_score": flicker_score,
            "motion_magnitude": motion_mag, "motion_score": motion_score,
            "sharpness_raw": sharp_raw, "sharpness_score": sharpness_score,
            "exposure_score": exposure_score,
        }

    # ---- report ---------------------------------------------------------------

    def score_chunk(self, chunk_id: str, shot_id: str, scene_id: str,
                    frames: Optional[torch.Tensor],
                    seam_similarity: Optional[float] = None,
                    identity_similarity: Optional[float] = None
                    ) -> QualityReport:
        report = QualityReport(chunk_id=chunk_id, shot_id=shot_id,
                               scene_id=scene_id)
        if frames is None:
            report.notes.append("no frames provided — dry-run/planning entry")
            report.recommended_action = "accept"
            return report

        m = self.measure(frames)
        report.measured = m
        report.flicker_score = m["flicker_score"]
        report.motion_score = m["motion_score"]
        report.sharpness_score = m["sharpness_score"]
        report.lighting_score = m["exposure_score"]
        if seam_similarity is not None:
            report.world_score = max(0.0, seam_similarity)
            report.measured["seam_similarity"] = seam_similarity
        if identity_similarity is not None:
            report.identity_score = max(0.0, identity_similarity)
        report.notes.append(
            "prompt_match/anatomy/cinematic scores await semantic validators "
            "(VBench/CLIP/face) — reported as 0.0, not faked")

        identity_min = self.thresholds.get("identity_threshold", 0.82)
        if identity_similarity is not None and identity_similarity < identity_min:
            report.recommended_action = "regenerate"
            report.notes.append(
                f"identity similarity {identity_similarity:.2f} < {identity_min}")
        elif m["flicker_score"] < 0.4:
            report.recommended_action = "stabilise"
        elif m["sharpness_score"] < 0.15 and m["exposure_score"] > 0.2:
            report.recommended_action = "detail_pass"
        else:
            report.recommended_action = "accept"
        return report
