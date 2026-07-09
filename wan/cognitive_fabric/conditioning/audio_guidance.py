"""Audio guidance (spec §18): parse and align audio to the shot timeline.
Phase 1: metadata + timing map; audio-to-expression conditioning is a later
phase behind the same AudioConditioningPack contract."""
from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AudioConditioningPack:
    audio_path: str
    transcript: str = ""
    phoneme_timing: List[Dict[str, Any]] = field(default_factory=list)
    beat_markers: List[float] = field(default_factory=list)
    emotion_curve: List[Dict[str, Any]] = field(default_factory=list)
    shot_alignment: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None


def probe_audio_duration(path: str) -> Optional[float]:
    p = Path(path)
    if not p.is_file():
        return None
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except (wave.Error, EOFError):
            return None
    return None  # non-wav probing arrives with the ffprobe integration


def build_audio_pack(path: str,
                     shot_ids: Optional[List[str]] = None) -> AudioConditioningPack:
    pack = AudioConditioningPack(audio_path=path)
    pack.duration_seconds = probe_audio_duration(path)
    if shot_ids and pack.duration_seconds:
        per_shot = pack.duration_seconds / len(shot_ids)
        pack.shot_alignment = {
            shot_id: {"start": i * per_shot, "end": (i + 1) * per_shot}
            for i, shot_id in enumerate(shot_ids)
        }
    return pack
