"""CUDA checker."""
from __future__ import annotations

from typing import Any, Dict

import torch


def check_cuda() -> Dict[str, Any]:
    available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "available": available,
        "torch_cuda_build": torch.version.cuda,
    }
    if available:
        info["devices"] = [{
            "name": torch.cuda.get_device_properties(i).name,
            "vram_gb": round(
                torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 1),
        } for i in range(torch.cuda.device_count())]
        info["bf16"] = torch.cuda.is_bf16_supported()
    else:
        info["note"] = ("no CUDA — real generation unavailable; fabric dry-run "
                        "and mock engine remain fully functional")
    return info
