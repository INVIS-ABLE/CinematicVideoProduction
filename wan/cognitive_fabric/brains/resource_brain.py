"""Resource Brain (spec §21): inspects real hardware and picks a hardware
mode. No fake numbers — unknown values are reported as None."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _ram_gb() -> Optional[float]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    try:
        import psutil  # optional
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        return None


class ResourceBrain:
    name = "resource"
    kind = "brain"

    def __init__(self, hardware_modes: Dict[str, Any] = None):
        self.hardware_modes = hardware_modes or {}

    def health_check(self) -> Dict[str, Any]:
        snap = self.snapshot()
        return {"status": "ok", "cuda": snap["cuda_available"],
                "mode": self.choose_hardware_mode()}

    def snapshot(self) -> Dict[str, Any]:
        cuda = torch.cuda.is_available()
        gpus = []
        if cuda:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024 ** 3, 1),
                    "capability": f"{props.major}.{props.minor}",
                })
        disk = shutil.disk_usage(".")
        return {
            "cuda_available": cuda,
            "gpus": gpus,
            "torch_version": torch.__version__,
            "bf16_supported": cuda and torch.cuda.is_bf16_supported(),
            "ram_gb": _ram_gb(),
            "disk_free_gb": round(disk.free / 1024 ** 3, 1),
        }

    def choose_hardware_mode(self) -> str:
        snap = self.snapshot()
        if not snap["cuda_available"]:
            return "low_vram"
        vram = max((g["vram_gb"] for g in snap["gpus"]), default=0)
        if len(snap["gpus"]) > 1 and vram * len(snap["gpus"]) >= 40:
            return "multi_gpu"
        if vram >= 22:
            return "studio_4090"
        if vram >= 12:
            return "creator"
        return "low_vram"

    def mode_settings(self) -> Dict[str, Any]:
        return dict(self.hardware_modes.get(self.choose_hardware_mode(), {}))
