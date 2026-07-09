"""Sound Brain (spec §18): audio ingest + shot alignment interface.
Local STT / lip-sync conditioning are later phases behind the same
AudioConditioningPack contract."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..conditioning.audio_guidance import AudioConditioningPack, build_audio_pack


class SoundBrain:
    name = "sound"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "metadata+alignment",
                "planned": ["local STT", "beat detection", "lip-sync repair"]}

    def align_audio(self, audio_path: str,
                    shot_ids: List[str]) -> AudioConditioningPack:
        return build_audio_pack(audio_path, shot_ids)
