"""Fabric runtime: the single object generate.py talks to.

Wires config → registry → brains → memories → orchestrators, selects the
engine (real Wan TI2V when CUDA + checkpoint are present, mock otherwise —
always saying which one it picked), and exposes the three cognitive tasks:

    run_short(prompt, …)          — cognitive-short
    run_film(storyboard, …)       — cognitive-film
    run_anime_episode(…)          — cognitive-anime-episode
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .brains.director_brain import DirectorBrain
from .brains.resource_brain import ResourceBrain
from .fabric_bus import FabricBus
from .fabric_config import FabricConfig, load_config
from .fabric_hooks import attention_backend_report
from .fabric_logging import get_fabric_logger
from .fabric_registry import FabricRegistry
from .pipeline.generation_controller import MockWanEngine
from .pipeline.longform_orchestrator import LongformOrchestrator
from .pipeline.storyboard_orchestrator import StoryboardOrchestrator

logger = get_fabric_logger("runtime")


class FabricRuntime:
    def __init__(self, config: Optional[FabricConfig] = None):
        self.config = config or FabricConfig()
        self.bus = FabricBus()
        self.registry = FabricRegistry()
        self.director = DirectorBrain()
        self.resources = ResourceBrain(self.config.data.get("hardware_modes"))
        self.orchestrator = StoryboardOrchestrator(self.config)
        self.longform = LongformOrchestrator(self.config)
        self.registry.register("director", self.director, "brain")
        self.registry.register("resources", self.resources, "brain")
        self.registry.register("orchestrator", self.orchestrator, "pipeline")

    @classmethod
    def from_config_path(cls, path: Optional[str] = None,
                         overrides: Optional[Dict[str, Any]] = None
                         ) -> "FabricRuntime":
        return cls(load_config(path, overrides))

    # ------------------------------------------------------------------
    # engine selection — explicit, never silent
    # ------------------------------------------------------------------

    def select_engine(self, ckpt_dir: Optional[str],
                      force_mock: bool = False) -> Any:
        import torch
        report = attention_backend_report()
        if force_mock or ckpt_dir is None or not torch.cuda.is_available():
            reason = ("forced" if force_mock else
                      "no --ckpt_dir given" if ckpt_dir is None else
                      "CUDA unavailable")
            logger.warning("engine: MOCK (%s). Real Wan generation needs a "
                           "checkpoint dir + CUDA GPU. attention=%s",
                           reason, report["selected_backend"])
            return MockWanEngine()
        from .pipeline.generation_controller import WanTI2VEngine
        logger.info("engine: Wan2.2 TI2V-5B from %s (attention=%s)",
                    ckpt_dir, report["selected_backend"])
        return WanTI2VEngine(ckpt_dir)

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------

    def run_short(self, prompt: str, *, project_dir: str,
                  duration_seconds: float = 30.0,
                  negative_prompt: str = "",
                  style: str = "ultra realistic cinematic",
                  ckpt_dir: Optional[str] = None,
                  dry_run: bool = False,
                  force_mock: bool = False,
                  first_frame_image: Optional[str] = None,
                  reference_images: Optional[list] = None) -> Dict[str, Any]:
        cf = self.config.data["cognitive_fabric"]
        storyboard = self.director.plan_from_prompt(
            prompt, duration_seconds=duration_seconds, style=style,
            negative_prompt=negative_prompt,
            anime_mode=cf["anime_mode_enabled"],
            chunk_seconds=cf["chunk_seconds"])
        if first_frame_image:
            if not Path(first_frame_image).is_file():
                raise FileNotFoundError(
                    f"--image not found: {first_frame_image}")
            storyboard["project"]["first_frame_image"] = first_frame_image
        if reference_images:
            missing = [r for r in reference_images if not Path(r).is_file()]
            if missing:
                raise FileNotFoundError(f"reference images not found: {missing}")
            storyboard["reference_uploads"] = [
                {"path": r, "role": "auto"} for r in reference_images]
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        (Path(project_dir) / "storyboard.json").write_text(
            json.dumps(storyboard, indent=2), encoding="utf-8")
        if dry_run:
            return self.orchestrator.dry_run(storyboard)
        engine = self.select_engine(ckpt_dir, force_mock)
        return self.orchestrator.run(storyboard, project_dir, engine)

    def run_film(self, storyboard_path: str, *, project_dir: str,
                 ckpt_dir: Optional[str] = None, dry_run: bool = False,
                 force_mock: bool = False) -> Dict[str, Any]:
        if dry_run:
            return self.longform.dry_run(storyboard_path, project_dir)
        engine = self.select_engine(ckpt_dir, force_mock)
        return self.longform.run(storyboard_path, project_dir, engine)

    def run_anime_episode(self, series_bible_path: str, episode_path: str, *,
                          project_dir: str, ckpt_dir: Optional[str] = None,
                          dry_run: bool = False,
                          force_mock: bool = False) -> Dict[str, Any]:
        from .brains.anime_brain import SeriesBible
        from .pipeline.anime_episode_pipeline import AnimeEpisodePipeline
        bible = SeriesBible(**json.loads(
            Path(series_bible_path).read_text(encoding="utf-8")))
        pipeline = AnimeEpisodePipeline(self.config)
        if dry_run:
            return pipeline.orchestrator.dry_run(episode_path)
        engine = self.select_engine(ckpt_dir, force_mock)
        episode_id = Path(episode_path).stem
        return pipeline.run_episode(bible, episode_path, project_dir, engine,
                                    episode_id)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "fabric_version": __import__(
                "wan.cognitive_fabric", fromlist=["__version__"]).__version__,
            "attention": attention_backend_report(),
            "hardware": self.resources.snapshot(),
            "hardware_mode": self.resources.choose_hardware_mode(),
            "components": self.registry.health_check_all(),
        }
