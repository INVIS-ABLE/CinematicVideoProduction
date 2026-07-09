"""Director Brain: converts a raw idea (one prompt + settings) into a full
storyboard — acts, scenes, shots, camera/lighting/motion plans, continuity
rules. Deterministic heuristics in Phase 1 (same input → same plan, so the
whole pipeline is reproducible and testable); a local LLM planner can
replace `plan_from_prompt` later behind the same storyboard schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SHOT_CYCLE = [
    ("establishing wide", "crane_down", "24mm"),
    ("medium tracking", "tracking", "35mm"),
    ("close-up", "push_in", "85mm"),
    ("insert detail", "static", "50mm"),
    ("action", "handheld", "35mm"),
    ("reaction", "static", "85mm"),
    ("final hero shot", "dolly_in", "35mm"),
]

_LOCATION_TABLE = [
    ("city", "urban cityscape"), ("castle", "ruined castle"),
    ("forest", "dense forest"), ("desert", "open desert"),
    ("ocean", "open ocean"), ("beach", "coastal beach"),
    ("mountain", "mountain range"), ("space", "deep space"),
    ("street", "city street"), ("rooftop", "high rooftop"),
    ("warehouse", "industrial warehouse"), ("village", "rural village"),
    ("construction", "construction site"), ("office", "modern interior"),
]

_TIME_TABLE = [
    ("night", "night"), ("sunset", "golden hour"), ("golden hour", "golden hour"),
    ("dawn", "dawn"), ("sunrise", "dawn"), ("dusk", "blue hour"),
    ("morning", "morning"), ("noon", "midday"),
]

_WEATHER_TABLE = [
    ("rain", "rain"), ("storm", "storm"), ("snow", "snow"), ("fog", "fog"),
    ("mist", "mist"), ("overcast", "overcast"), ("wind", "wind"),
]

_LIGHTING_BY_TIME = {
    "night": {"key": "practical neon and moonlight", "mood": "low-key",
              "direction": "motivated by practicals",
              "colour_temperature": "cool with warm accents"},
    "golden hour": {"key": "low warm sun", "mood": "romantic",
                    "direction": "side-lit", "colour_temperature": "warm"},
    "dawn": {"key": "soft cold skylight", "mood": "quiet",
             "direction": "east side", "colour_temperature": "cool"},
    "blue hour": {"key": "deep blue ambient", "mood": "melancholic",
                  "direction": "ambient", "colour_temperature": "cold"},
    "day": {"key": "natural daylight", "mood": "balanced",
            "direction": "high side", "colour_temperature": "neutral"},
    "midday": {"key": "hard top sun", "mood": "harsh",
               "direction": "top-down", "colour_temperature": "neutral"},
    "morning": {"key": "clean morning sun", "mood": "fresh",
                "direction": "low side", "colour_temperature": "slightly warm"},
}


def _lookup(text: str, table: List[tuple], fallback: str) -> str:
    lower = text.lower()
    for needle, value in table:
        if needle in lower:
            return value
    return fallback


class DirectorBrain:
    name = "director"
    kind = "brain"

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "planner": "deterministic-heuristic-v1"}

    def plan_from_prompt(self, prompt: str, *,
                         duration_seconds: float = 30.0,
                         style: str = "cinematic",
                         negative_prompt: str = "",
                         anime_mode: bool = False,
                         chunk_seconds: float = 5.0,
                         title: Optional[str] = None) -> Dict[str, Any]:
        """One idea → storyboard dict matching configs/storyboard_schema.json."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("director brain needs a non-empty prompt")

        location = _lookup(prompt, _LOCATION_TABLE, "location from prompt")
        time_of_day = _lookup(prompt, _TIME_TABLE, "day")
        weather = _lookup(prompt, _WEATHER_TABLE, "clear")
        lighting = _LIGHTING_BY_TIME.get(time_of_day, _LIGHTING_BY_TIME["day"])

        # bound shots: >= 1, each shot 1 chunk long in short mode,
        # capped so per-shot duration stays >= 2s
        shot_count = max(1, min(8, int(round(duration_seconds / chunk_seconds))))
        per_shot = duration_seconds / shot_count

        world_id = "world_001"
        characters = self._extract_characters(prompt)
        shots = []
        for i in range(shot_count):
            shot_type, camera_move, lens = SHOT_CYCLE[i % len(SHOT_CYCLE)]
            shots.append({
                "shot_id": f"shot_{i + 1:03d}",
                "scene_id": "scene_001",
                "duration_seconds": round(per_shot, 2),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "shot_type": shot_type,
                "character_ids": [c["character_id"] for c in characters],
                "object_ids": [],
                "world_id": world_id,
                "camera_plan": {"movement": camera_move, "lens": lens,
                                "easing": "ease-in-out"},
                "lighting_plan": dict(lighting),
                "motion_plan": {"intensity": "moderate",
                                "subject_path": "consistent screen direction"},
                "physics_expectation": {},
                "transition_in": "cut" if i else "fade_in",
                "transition_out": "fade_out" if i == shot_count - 1 else "cut",
                "continuity_from_previous": i > 0,
                "continuity_to_next": i < shot_count - 1,
                "seed_strategy": "per_shot",
            })

        return {
            "project": {
                "title": title or (prompt[:48] + ("…" if len(prompt) > 48 else "")),
                "duration_seconds": duration_seconds,
                "style": style,
                "anime_mode": anime_mode,
                "aspect_ratio": "16:9",
                "frame_rate": 24,
            },
            "characters": characters,
            "objects": [],
            "worlds": [{
                "world_id": world_id,
                "description": f"{location}, {time_of_day}, {weather}",
                "location_type": location,
                "time_of_day": time_of_day,
                "weather": weather,
                "light_direction": lighting["direction"],
            }],
            "acts": [{"act_id": "act_001", "scenes": ["scene_001"]}],
            "scenes": [{
                "scene_id": "scene_001",
                "act_id": "act_001",
                "world_id": world_id,
                "purpose": "realise the user's idea as a coherent sequence",
                "emotional_beat": _lookup(prompt, [
                    ("emotional", "intimate"), ("epic", "awe"),
                    ("dark", "dread"), ("hopeful", "uplift"),
                    ("tense", "suspense")], "engagement"),
                "shots": [s["shot_id"] for s in shots],
            }],
            "shots": shots,
        }

    @staticmethod
    def _extract_characters(prompt: str) -> List[Dict[str, Any]]:
        """Heuristic single-subject extraction; a local VLM upgrades this."""
        lower = prompt.lower()
        subjects = [
            ("knight", "weathered knight"), ("hero", "the hero"),
            ("woman", "the woman"), ("man", "the man"), ("girl", "the girl"),
            ("boy", "the boy"), ("warrior", "the warrior"),
            ("detective", "the detective"), ("astronaut", "the astronaut"),
            ("samurai", "the samurai"), ("robot", "the robot"),
            ("cat", "the cat"), ("dog", "the dog"),
        ]
        characters = []
        for i, (needle, name) in enumerate(subjects):
            if needle in lower:
                characters.append({
                    "character_id": f"char_{len(characters) + 1:03d}",
                    "name": name,
                    "source": "prompt",
                })
                if len(characters) >= 3:
                    break
        return characters
