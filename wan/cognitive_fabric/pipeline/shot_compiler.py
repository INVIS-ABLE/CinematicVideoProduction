"""Shot compiler: applies cinematographer/lighting/motion brains to fill any
missing plan fields on a shot before chunking."""
from __future__ import annotations

from typing import Any, Dict

from ..brains.cinematographer_brain import CinematographerBrain
from ..brains.lighting_brain import LightingBrain
from ..brains.motion_brain import MotionBrain


def compile_shot(shot: Dict[str, Any],
                 world: Dict[str, Any] = None) -> Dict[str, Any]:
    out = dict(shot)
    shot_type = out.get("shot_type", "medium")
    if not out.get("camera_plan"):
        out["camera_plan"] = CinematographerBrain().plan_camera(shot_type)
    if not out.get("lighting_plan"):
        world = world or {}
        out["lighting_plan"] = LightingBrain().plan_lighting(
            world.get("time_of_day", "day"), world.get("weather", "clear"))
    if not out.get("motion_plan"):
        out["motion_plan"] = MotionBrain().plan_motion(shot_type)
    return out
