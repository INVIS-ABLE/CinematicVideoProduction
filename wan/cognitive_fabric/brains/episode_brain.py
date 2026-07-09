"""Episode Brain (spec §14) — episode/series continuity interface.
Persists series memory across projects; full episode pipeline is Phase 7."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .anime_brain import SeriesBible


class EpisodeBrain:
    name = "episode"
    kind = "brain"

    def __init__(self, series_root: str = "projects/series"):
        self.series_root = Path(series_root)

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "series_root": str(self.series_root)}

    def save_series_bible(self, bible: SeriesBible) -> Path:
        d = self.series_root / bible.series_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "series_bible.json"
        path.write_text(json.dumps(asdict(bible), indent=2), encoding="utf-8")
        return path

    def load_series_bible(self, series_id: str) -> Optional[SeriesBible]:
        path = self.series_root / series_id / "series_bible.json"
        if not path.is_file():
            return None
        return SeriesBible(**json.loads(path.read_text(encoding="utf-8")))

    def record_episode(self, series_id: str, episode_id: str) -> None:
        bible = self.load_series_bible(series_id)
        if bible is None:
            raise KeyError(f"unknown series: {series_id}")
        if episode_id not in bible.episode_history:
            bible.episode_history.append(episode_id)
        self.save_series_bible(bible)
