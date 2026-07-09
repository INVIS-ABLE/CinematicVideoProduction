"""Reference Brain: routes user uploads into the reference memory and the
token stack with sensible role priorities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..conditioning.reference_token_stack import ReferenceTokenStack
from ..memory.reference_memory import ReferenceMemory

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


class ReferenceBrain:
    name = "reference"
    kind = "brain"

    def __init__(self, memory: ReferenceMemory, stack: ReferenceTokenStack):
        self.memory = memory
        self.stack = stack

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "references": len(self.memory.references),
                "tokens": len(self.stack)}

    def ingest(self, path: str, role: str = "auto") -> Dict[str, Any]:
        suffix = Path(path).suffix.lower()
        if role == "auto":
            role = ("image" if suffix in _IMAGE_EXTS else
                    "video" if suffix in _VIDEO_EXTS else
                    "audio" if suffix in _AUDIO_EXTS else "image")
        record = self.memory.register(path, role)
        if role in ("image", "style", "world", "lighting"):
            self.stack.add_image_reference(path)
        elif role == "video":
            self.stack.add_video_reference(path)
        elif role == "audio":
            self.stack.add_audio_reference(path)
        elif role == "character":
            self.stack.add_character_reference(path, record["ref_id"])
        return record
