"""Long-form orchestrator (spec §15): Film → Acts → Scenes → Shots → Chunks.

Delegates scene execution to StoryboardOrchestrator; adds the film-level
hierarchy, dry-run planning for 30s..60min storyboards, and per-scene
export. Never holds the film in memory — every artifact streams to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..fabric_config import FabricConfig
from .storyboard_orchestrator import StoryboardOrchestrator


class LongformOrchestrator:
    def __init__(self, config: Optional[FabricConfig] = None):
        self.config = config or FabricConfig()
        self.scene_orchestrator = StoryboardOrchestrator(self.config)

    def film_plan(self, source: Any) -> Dict[str, Any]:
        sb = self.scene_orchestrator.load_storyboard(source)
        chunks = self.scene_orchestrator.plan(sb)
        chunks_by_scene: Dict[str, List[str]] = {}
        for c in chunks:
            chunks_by_scene.setdefault(c.scene_id, []).append(c.chunk_id)
        acts = sb.get("acts") or [{"act_id": "act_001",
                                   "scenes": [s["scene_id"]
                                              for s in sb.get("scenes", [])]}]
        fps = sb.get("project", {}).get("frame_rate", 24)
        total_frames = sum(c.frame_num for c in chunks)
        return {
            "film": sb["project"].get("title", "untitled"),
            "acts": [{
                "act_id": act["act_id"],
                "scenes": [{
                    "scene_id": scene_id,
                    "chunks": chunks_by_scene.get(scene_id, []),
                } for scene_id in act.get("scenes", [])],
            } for act in acts],
            "total_chunks": len(chunks),
            "total_frames": total_frames,
            "estimated_minutes": round(total_frames / fps / 60, 2),
            "max_minutes": self.config.get("cognitive_fabric.max_minutes", 60),
            # 2% headroom: 4n+1 frame snapping legitimately rounds each
            # shot's frame count up by a few frames
            "within_budget": total_frames / fps / 60 <=
                self.config.get("cognitive_fabric.max_minutes", 60) * 1.02,
        }

    def dry_run(self, source: Any, project_dir: str) -> Dict[str, Any]:
        plan = self.film_plan(source)
        out = Path(project_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "film_plan.json").write_text(json.dumps(plan, indent=2),
                                            encoding="utf-8")
        return plan

    def run(self, source: Any, project_dir: str, engine: Any) -> Dict[str, Any]:
        plan = self.dry_run(source, project_dir)
        report = self.scene_orchestrator.run(source, project_dir, engine)
        report["film_plan"] = str(Path(project_dir) / "film_plan.json")
        report["acts"] = len(plan["acts"])
        return report
