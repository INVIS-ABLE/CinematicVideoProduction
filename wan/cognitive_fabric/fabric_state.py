"""Persistent per-project fabric state: project tree, chunk ledger,
checkpointing, resume. A crash at minute 43 resumes from the last approved
chunk — never a full restart."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_SUBDIRS = [
    "assets", "scenes", "shots", "chunks", "frames_720p", "frames_1080p",
    "frames_1440p", "frames_4k", "audio", "subtitles", "exports", "logs",
    "checkpoints",
]


@dataclass
class ChunkLedgerEntry:
    chunk_id: str
    shot_id: str
    scene_id: str
    status: str = "planned"
    attempt: int = 0
    output_path: Optional[str] = None
    quality_overall: float = 0.0
    saved_frames: int = 0
    trimmed_lead_frames: int = 0
    trimmed_tail_frames: int = 0
    updated_at: float = field(default_factory=time.time)


class FabricState:
    def __init__(self, project_id: str, project_dir: str):
        self.project_id = project_id
        self.project_dir = Path(project_dir)
        self.storyboard: Dict[str, Any] = {}
        self.ledger: Dict[str, ChunkLedgerEntry] = {}
        self.run_meta: Dict[str, Any] = {"created_at": time.time()}

    # ---- project tree -----------------------------------------------------

    def ensure_project_tree(self) -> Path:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        for sub in PROJECT_SUBDIRS:
            (self.project_dir / sub).mkdir(exist_ok=True)
        return self.project_dir

    # ---- ledger ------------------------------------------------------------

    def register_chunk(self, chunk_id: str, shot_id: str, scene_id: str) -> None:
        if chunk_id not in self.ledger:
            self.ledger[chunk_id] = ChunkLedgerEntry(
                chunk_id=chunk_id, shot_id=shot_id, scene_id=scene_id)

    def update_chunk(self, chunk_id: str, **fields: Any) -> None:
        entry = self.ledger[chunk_id]
        for key, value in fields.items():
            setattr(entry, key, value)
        entry.updated_at = time.time()

    def pending_chunks(self) -> List[str]:
        order = list(self.ledger.keys())
        return [c for c in order
                if self.ledger[c].status not in ("approved", "accepted_with_warning")]

    def approved_chunks(self) -> List[str]:
        return [c for c, e in self.ledger.items()
                if e.status in ("approved", "accepted_with_warning")]

    # ---- checkpoint / resume ------------------------------------------------

    def checkpoint_path(self) -> Path:
        return self.project_dir / "checkpoints" / "fabric_state.json"

    def save_checkpoint(self) -> Path:
        self.ensure_project_tree()
        path = self.checkpoint_path()
        payload = {
            "project_id": self.project_id,
            "storyboard": self.storyboard,
            "ledger": {k: asdict(v) for k, v in self.ledger.items()},
            "run_meta": self.run_meta,
            "saved_at": time.time(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX & Windows same-volume
        return path

    @classmethod
    def load_checkpoint(cls, project_dir: str) -> "FabricState":
        path = Path(project_dir) / "checkpoints" / "fabric_state.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = cls(payload["project_id"], project_dir)
        state.storyboard = payload.get("storyboard", {})
        state.run_meta = payload.get("run_meta", {})
        for chunk_id, entry in payload.get("ledger", {}).items():
            state.ledger[chunk_id] = ChunkLedgerEntry(**entry)
        return state

    @classmethod
    def resume_or_create(cls, project_id: str, project_dir: str) -> "FabricState":
        path = Path(project_dir) / "checkpoints" / "fabric_state.json"
        if path.is_file():
            state = cls.load_checkpoint(project_dir)
            state.run_meta["resumed_at"] = time.time()
            return state
        state = cls(project_id, project_dir)
        state.ensure_project_tree()
        return state
