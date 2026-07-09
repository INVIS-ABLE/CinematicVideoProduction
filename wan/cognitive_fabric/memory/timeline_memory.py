"""Timeline memory — ordered record of approved chunks/shots/scenes, the
source of truth for stitching and export (a lightweight internal EDL)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TimelineMemory:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def append_clip(self, *, chunk_id: str, shot_id: str, scene_id: str,
                    path: str, frame_num: int, fps: int,
                    transition_in: str = "cut",
                    transition_out: str = "cut",
                    trimmed_lead_frames: int = 0,
                    trimmed_tail_frames: int = 0) -> None:
        self.entries.append({
            "chunk_id": chunk_id, "shot_id": shot_id, "scene_id": scene_id,
            "path": path, "frame_num": frame_num, "fps": fps,
            "transition_in": transition_in, "transition_out": transition_out,
            "trimmed_lead_frames": trimmed_lead_frames,
            "trimmed_tail_frames": trimmed_tail_frames,
        })

    def clips_for_scene(self, scene_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e["scene_id"] == scene_id]

    def total_frames(self) -> int:
        return sum(e["frame_num"] for e in self.entries)

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"timeline": self.entries}, indent=2),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str) -> "TimelineMemory":
        timeline = cls()
        p = Path(path)
        if p.is_file():
            timeline.entries = json.loads(
                p.read_text(encoding="utf-8")).get("timeline", [])
        return timeline
