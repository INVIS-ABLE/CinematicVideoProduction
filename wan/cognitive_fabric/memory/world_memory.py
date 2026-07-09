"""World Memory Engine (spec §8): tracks scene-consistency state and detects
world drift between shots. Conditioning output is prompt clauses today and
world tokens (via the reference encoder) for Phase 4 attention injection."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# fields that must not silently change while the storyboard stays in-scene
CONTINUITY_CRITICAL = (
    "location_type", "time_of_day", "weather", "light_direction",
    "architecture", "scale",
)


@dataclass
class WorldState:
    world_id: str
    description: str = ""
    location_type: str = "unspecified"
    time_of_day: str = "day"
    weather: str = "clear"
    season: str = "unspecified"
    scale: str = "human"
    architecture: str = ""
    light_direction: str = "unspecified"
    layout_graph: Dict[str, Any] = field(default_factory=dict)
    lighting_state: Dict[str, Any] = field(default_factory=dict)
    weather_state: Dict[str, Any] = field(default_factory=dict)
    material_state: Dict[str, Any] = field(default_factory=dict)
    physics_state: Dict[str, Any] = field(default_factory=dict)
    continuity_constraints: List[str] = field(default_factory=list)
    reference_tokens: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def prompt_clauses(self) -> List[str]:
        clauses = []
        if self.description:
            clauses.append(f"consistent world: {self.description}")
        clauses.append(
            f"same {self.location_type} location, {self.time_of_day}, "
            f"{self.weather} weather throughout")
        if self.light_direction != "unspecified":
            clauses.append(f"light direction stays {self.light_direction}")
        clauses.extend(self.continuity_constraints)
        return clauses


class WorldMemory:
    def __init__(self):
        self.worlds: Dict[str, WorldState] = {}

    def create_world(self, world_id: str, **kwargs: Any) -> WorldState:
        state = WorldState(world_id=world_id, **kwargs)
        self.worlds[world_id] = state
        return state

    def get(self, world_id: str) -> Optional[WorldState]:
        return self.worlds.get(world_id)

    def update_world_from_shot(self, world_id: str,
                               shot_observations: Dict[str, Any]) -> WorldState:
        state = self.worlds[world_id]
        for key, value in shot_observations.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.layout_graph[key] = value
        state.updated_at = time.time()
        return state

    def get_world_conditioning(self, world_id: str) -> List[str]:
        state = self.worlds.get(world_id)
        return state.prompt_clauses() if state else []

    def detect_world_drift(self, world_id: str,
                           observed: Dict[str, Any]) -> List[Dict[str, str]]:
        """Compare an observed shot description against stored world state.
        Returns a list of drift issues (empty = consistent)."""
        state = self.worlds.get(world_id)
        if state is None:
            return []
        issues = []
        for key in CONTINUITY_CRITICAL:
            expected = getattr(state, key, None)
            seen = observed.get(key)
            if seen is None or expected in (None, "", "unspecified"):
                continue
            if str(seen).lower() != str(expected).lower():
                issues.append({
                    "field": key,
                    "expected": str(expected),
                    "observed": str(seen),
                    "severity": "drift",
                })
        return issues

    # ---- persistence ------------------------------------------------------

    def save(self, directory: str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "world_memory.json"
        path.write_text(json.dumps(
            {wid: asdict(w) for wid, w in self.worlds.items()}, indent=2),
            encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str) -> "WorldMemory":
        path = Path(directory) / "world_memory.json"
        memory = cls()
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for wid, data in payload.items():
                memory.worlds[wid] = WorldState(**data)
        return memory
