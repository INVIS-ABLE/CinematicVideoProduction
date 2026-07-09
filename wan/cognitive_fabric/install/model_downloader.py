"""Model downloader: verify-first, local-only by default (spec §22).
Downloads only run when the user explicitly enables network fetch; they use
huggingface_hub against the official open Wan-AI repos."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

OFFICIAL_REPOS = {
    "ti2v-5B": "Wan-AI/Wan2.2-TI2V-5B",
    "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B",
    "i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B",
    "s2v-14B": "Wan-AI/Wan2.2-S2V-14B",
    "animate-14B": "Wan-AI/Wan2.2-Animate-14B",
}

EXPECTED_FILES = {
    "ti2v-5B": ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"],
    "t2v-A14B": ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"],
}


def verify_local_model(task: str, ckpt_dir: str) -> Dict[str, Any]:
    root = Path(ckpt_dir)
    if not root.is_dir():
        return {"task": task, "present": False,
                "note": f"checkpoint dir missing: {ckpt_dir}"}
    expected = EXPECTED_FILES.get(task, [])
    missing = [f for f in expected if not (root / f).is_file()]
    safetensors = list(root.rglob("*.safetensors"))
    return {
        "task": task,
        "present": not missing and bool(safetensors),
        "missing_files": missing,
        "safetensors_count": len(safetensors),
    }


def download_model(task: str, dest_dir: str,
                   allow_network: bool = False) -> Dict[str, Any]:
    if not allow_network:
        return {"task": task, "downloaded": False,
                "note": ("local-only mode: pass allow_network=True (or run "
                         "first_run_setup with --download) to fetch "
                         f"{OFFICIAL_REPOS.get(task)} from Hugging Face")}
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {"task": task, "downloaded": False,
                "note": "pip install huggingface_hub to enable downloads"}
    repo = OFFICIAL_REPOS[task]
    path = snapshot_download(repo_id=repo, local_dir=dest_dir)
    return {"task": task, "downloaded": True, "path": path}
