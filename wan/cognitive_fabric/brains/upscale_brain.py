"""Upscale Brain: decides the progressive-resolution route for a project
given hardware mode and target output; execution lives in
pipeline/hierarchical_scaler.py."""
from __future__ import annotations

from typing import Any, Dict, List

_STAGE_ORDER = ["base", "hd", "qhd", "uhd"]
_STAGE_BY_NAME = {"720p": "base", "1080p": "hd", "1440p": "qhd",
                  "4k": "uhd", "2160p": "uhd", "480p": "base"}


class UpscaleBrain:
    name = "upscale"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok",
                "backends": ["torch-interpolate (native)",
                             "esrgan (hook)", "rife (hook)"]}

    def plan_stages(self, base: str, final: str) -> List[str]:
        start = _STAGE_BY_NAME.get(base, "base")
        end = _STAGE_BY_NAME.get(final, "uhd")
        s, e = _STAGE_ORDER.index(start), _STAGE_ORDER.index(end)
        if e <= s:
            return []
        return _STAGE_ORDER[s + 1:e + 1]
