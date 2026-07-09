"""Continuity Brain: checks each chunk/shot against project memory and the
previous chunk; produces continuity clauses for the next prompt."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..memory.temporal_kv_cache import TemporalKVCache
from ..memory.world_memory import WorldMemory

TRANSITION_KINDS = ("cut", "hard_cut", "soft_cut", "match_cut", "continuation",
                    "time_jump", "location_change", "fade_in", "fade_out",
                    "dissolve")


class ContinuityBrain:
    name = "continuity"
    kind = "brain"

    def __init__(self, worlds: WorldMemory, cache: TemporalKVCache,
                 seam_threshold: float = 0.5):
        self.worlds = worlds
        self.cache = cache
        self.seam_threshold = seam_threshold

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "cached_chunks": len(self.cache)}

    def continuity_clauses(self, shot: Dict[str, Any],
                           previous_shot: Optional[Dict[str, Any]]) -> List[str]:
        clauses: List[str] = []
        if previous_shot is not None and shot.get("continuity_from_previous"):
            clauses.append(
                "direct continuation of the previous shot: same location, "
                "same lighting, same weather, same wardrobe, same props")
            prev_cam = (previous_shot.get("camera_plan") or {}).get("movement")
            if prev_cam:
                clauses.append(
                    f"screen direction consistent with previous {prev_cam} move")
        world_id = shot.get("world_id")
        if world_id:
            clauses.extend(self.worlds.get_world_conditioning(world_id))
        return clauses

    def classify_transition(self, shot: Dict[str, Any],
                            previous_shot: Optional[Dict[str, Any]]) -> str:
        if previous_shot is None:
            return "fade_in"
        if shot.get("world_id") != previous_shot.get("world_id"):
            return "location_change"
        if not shot.get("continuity_from_previous", True):
            return "hard_cut"
        return "continuation"

    def check_seam(self, opening_frames) -> Dict[str, Any]:
        """Compare a chunk's opening frames with the cached previous terminal
        embedding. Returns {similarity, ok} (ok=None when nothing cached)."""
        similarity = self.cache.continuity_similarity(opening_frames)
        if similarity is None:
            return {"similarity": None, "ok": None,
                    "note": "no previous chunk cached"}
        return {"similarity": similarity,
                "ok": similarity >= self.seam_threshold,
                "note": f"threshold {self.seam_threshold}"}
