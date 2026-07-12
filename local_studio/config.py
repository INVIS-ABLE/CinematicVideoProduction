"""Configuration and hard product limits for the local cinematic studio."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

MAX_IMAGES = 15
MAX_VIDEOS = 3
MAX_VOICE_SAMPLES = 3
MIN_DURATION_SECONDS = 4
MAX_DURATION_SECONDS = 15

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
VOICE_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

MAX_UPLOAD_BYTES = {
    "image": 64 * 1024 * 1024,
    "video": 2 * 1024 * 1024 * 1024,
    "voice": 256 * 1024 * 1024,
}

ASPECT_RATIOS: Dict[str, float] = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:5": 4 / 5,
    "2.35:1": 2.35,
}

# Native-area targets chosen to stay close to Wan TI2V's supported grids.
NATIVE_AREAS = {
    "480p": 832 * 480,
    "720p": 1280 * 704,
}


def _truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def native_size(resolution: str, aspect_ratio: str, multiple: int = 32) -> Tuple[int, int]:
    """Return a Wan-safe (width, height) under the selected native pixel area."""
    if resolution not in NATIVE_AREAS:
        raise ValueError(f"unsupported resolution: {resolution}")
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"unsupported aspect ratio: {aspect_ratio}")

    area = NATIVE_AREAS[resolution]
    ratio = ASPECT_RATIOS[aspect_ratio]
    height = int((area / ratio) ** 0.5)
    width = int(height * ratio)
    width = max(multiple, width // multiple * multiple)
    height = max(multiple, height // multiple * multiple)

    while width * height > area:
        if width / height >= ratio:
            width -= multiple
        else:
            height -= multiple
    return width, height


@dataclass(frozen=True)
class StudioSettings:
    """Runtime settings. Every path is local and no network access is required."""

    root_dir: Path
    ti2v_checkpoint: Optional[Path]
    s2v_checkpoint: Optional[Path]
    host: str = "127.0.0.1"
    port: int = 7860
    force_mock: bool = False
    local_only: bool = True
    max_workers: int = 1

    @classmethod
    def from_env(cls, repo_root: Optional[Path] = None) -> "StudioSettings":
        base = (repo_root or Path.cwd()).resolve()
        root = Path(os.getenv("CINESTUDIO_PROJECTS_DIR", base / "projects" / "studio"))
        ti2v_raw = os.getenv("CINESTUDIO_TI2V_CKPT")
        s2v_raw = os.getenv("CINESTUDIO_S2V_CKPT")
        return cls(
            root_dir=root.expanduser().resolve(),
            ti2v_checkpoint=Path(ti2v_raw).expanduser().resolve() if ti2v_raw else None,
            s2v_checkpoint=Path(s2v_raw).expanduser().resolve() if s2v_raw else None,
            host=os.getenv("CINESTUDIO_HOST", "127.0.0.1"),
            port=int(os.getenv("CINESTUDIO_PORT", "7860")),
            force_mock=_truthy(os.getenv("CINESTUDIO_FORCE_MOCK"), False),
            local_only=_truthy(os.getenv("CINESTUDIO_LOCAL_ONLY"), True),
            max_workers=max(1, int(os.getenv("CINESTUDIO_MAX_WORKERS", "1"))),
        )

    def checkpoint_exists(self, kind: str) -> bool:
        path = self.ti2v_checkpoint if kind == "ti2v" else self.s2v_checkpoint
        return bool(path and path.is_dir())
