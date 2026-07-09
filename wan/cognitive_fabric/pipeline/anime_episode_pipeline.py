"""Anime episode pipeline (spec §14) — Phase 1: series-bible-aware wrapper
around the storyboard orchestrator. Anime-specific validators/timing land in
Phase 7; the episode contract and series persistence work today."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..brains.anime_brain import AnimeBrain, SeriesBible
from ..brains.episode_brain import EpisodeBrain
from ..fabric_config import FabricConfig
from .storyboard_orchestrator import StoryboardOrchestrator


class AnimeEpisodePipeline:
    def __init__(self, config: Optional[FabricConfig] = None,
                 series_root: str = "projects/series"):
        self.config = config or FabricConfig()
        self.anime = AnimeBrain()
        self.episodes = EpisodeBrain(series_root)
        self.orchestrator = StoryboardOrchestrator(self.config)

    def run_episode(self, series_bible: SeriesBible, episode_storyboard: Any,
                    project_dir: str, engine: Any,
                    episode_id: str) -> Dict[str, Any]:
        storyboard = self.orchestrator.load_storyboard(episode_storyboard)
        clauses = self.anime.anime_prompt_clauses(series_bible.visual_style)
        for shot in storyboard["shots"]:
            shot["prompt"] = shot["prompt"] + ". " + ". ".join(clauses)
        self.episodes.save_series_bible(series_bible)
        report = self.orchestrator.run(storyboard, project_dir, engine,
                                       project_id=episode_id)
        self.episodes.record_episode(series_bible.series_id, episode_id)
        report["series_id"] = series_bible.series_id
        return report
