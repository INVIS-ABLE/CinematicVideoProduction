"""Dependency checker: verifies the local environment honestly — every check
reports found/missing/version, nothing is auto-installed from here."""
from __future__ import annotations

import importlib
import shutil
import sys
from typing import Any, Dict

BASE_MODULES = ["torch", "torchvision", "easydict", "einops", "diffusers",
                "transformers", "tokenizers", "PIL", "numpy", "tqdm", "ftfy"]
OPTIONAL_MODULES = ["flash_attn", "imageio", "yaml", "librosa", "decord",
                    "cv2", "peft"]


def check_python() -> Dict[str, Any]:
    ok = sys.version_info >= (3, 9)
    return {"found": True, "version": sys.version.split()[0], "ok": ok}


def check_module(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return {"found": True, "version": version}
    except Exception as e:
        return {"found": False, "error": str(e)[:200]}


def check_binary(name: str) -> Dict[str, Any]:
    path = shutil.which(name)
    return {"found": path is not None, "path": path}


def check_all() -> Dict[str, Any]:
    return {
        "python": check_python(),
        "modules": {m: check_module(m) for m in BASE_MODULES},
        "optional_modules": {m: check_module(m) for m in OPTIONAL_MODULES},
        "binaries": {b: check_binary(b) for b in ("ffmpeg", "git")},
    }
