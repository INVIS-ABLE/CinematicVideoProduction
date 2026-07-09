"""Scene compiler: fills scene-level defaults and orders shots for the
orchestrator (emotional beat pacing hooks land with the editor brain)."""
from __future__ import annotations

from typing import Any, Dict, List


def compile_scenes(storyboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenes = []
    shots_by_id = {s["shot_id"]: s for s in storyboard.get("shots", [])}
    for scene in storyboard.get("scenes", []):
        ordered = [shots_by_id[sid] for sid in scene.get("shots", [])
                   if sid in shots_by_id]
        scenes.append({**scene, "shot_specs": ordered,
                       "duration_seconds": sum(
                           s["duration_seconds"] for s in ordered)})
    return scenes
