"""Object memory — domain wrapper over the Global Identity Matrix for props,
vehicles, weapons and other continuity-critical objects."""
from __future__ import annotations

from typing import Any, Dict, List

from .global_identity_matrix import GlobalIdentityMatrix


class ObjectMemory:
    def __init__(self, matrix: GlobalIdentityMatrix):
        self.matrix = matrix

    def register(self, object_id: str, name: str, **kwargs: Any) -> Dict[str, Any]:
        return self.matrix.register_object(object_id, name, **kwargs)

    def set_damage_state(self, object_id: str, damage_state: str) -> None:
        profile = self.matrix.objects.get(object_id)
        if profile is None:
            raise KeyError(f"unknown object: {object_id}")
        profile["damage_state"] = damage_state

    def prompt_clauses(self, object_ids: List[str]) -> List[str]:
        clauses = []
        for oid in object_ids:
            p = self.matrix.objects.get(oid)
            if not p:
                continue
            bits = [p["name"]]
            for key in ("colour", "material", "shape", "damage_state"):
                if p.get(key) and p[key] not in ("", "intact"):
                    bits.append(f"{key.replace('_', ' ')}: {p[key]}")
            clauses.append("consistent object: " + ", ".join(bits))
            if p.get("forbidden_changes"):
                clauses.append(
                    f"the {p['name']} must not change its {p['forbidden_changes']}")
        return clauses

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "objects": len(self.matrix.objects)}
