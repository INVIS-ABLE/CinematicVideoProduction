"""Lighting Brain: cinematic lighting plans with continuity locking."""
from __future__ import annotations

from typing import Any, Dict, Optional

_PRESETS = {
    "night_neon": {"key": "neon practicals", "rim": "cool blue rim",
                   "mood": "low-key", "direction": "motivated by signage",
                   "colour_temperature": "cool with warm accents"},
    "golden_hour": {"key": "low warm sun", "fill": "soft sky bounce",
                    "mood": "romantic", "direction": "side-lit",
                    "colour_temperature": "warm"},
    "overcast": {"key": "soft overcast skylight", "mood": "even",
                 "direction": "diffuse top", "colour_temperature": "neutral"},
    "firelight": {"key": "flickering warm firelight", "mood": "intimate",
                  "direction": "low motivated", "colour_temperature": "very warm"},
    "moonlight": {"key": "cold moonlight", "fill": "deep shadow",
                  "mood": "low-key", "direction": "high side",
                  "colour_temperature": "cold"},
}


class LightingBrain:
    name = "lighting"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "presets": list(_PRESETS)}

    def plan_lighting(self, time_of_day: str, weather: str,
                      user_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if time_of_day == "night":
            plan = dict(_PRESETS["night_neon"])
        elif time_of_day == "golden hour":
            plan = dict(_PRESETS["golden_hour"])
        elif weather in ("overcast", "rain", "storm"):
            plan = dict(_PRESETS["overcast"])
            if weather in ("rain", "storm"):
                plan["practicals"] = "wet-surface reflections of light sources"
        else:
            plan = {"key": "natural daylight", "mood": "balanced",
                    "direction": "high side", "colour_temperature": "neutral"}
        if user_plan:
            plan.update({k: v for k, v in user_plan.items() if v})
        return plan
