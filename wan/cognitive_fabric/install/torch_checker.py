"""Torch checker."""
from __future__ import annotations

from typing import Any, Dict

import torch


def check_torch() -> Dict[str, Any]:
    version = torch.__version__
    major, minor = (int(x) for x in version.split("+")[0].split(".")[:2])
    return {
        "found": True,
        "version": version,
        "meets_minimum": (major, minor) >= (2, 4),
        "minimum": "2.4.0 (upstream requirements.txt)",
    }
