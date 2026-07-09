"""Reference memory — registry of user-uploaded reference assets and their
assigned roles; feeds the reference token stack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReferenceMemory:
    def __init__(self):
        self.references: Dict[str, Dict[str, Any]] = {}

    def register(self, path: str, role: str, *,
                 scope: Optional[str] = None,
                 weight: float = 1.0, priority: int = 5) -> Dict[str, Any]:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        ref_id = f"ref_{digest}"
        self.references[ref_id] = {
            "ref_id": ref_id, "path": path, "role": role, "scope": scope,
            "weight": weight, "priority": priority,
        }
        return self.references[ref_id]

    def by_role(self, role: str) -> List[Dict[str, Any]]:
        return [r for r in self.references.values() if r["role"] == role]

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.references, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str) -> "ReferenceMemory":
        memory = cls()
        p = Path(path)
        if p.is_file():
            memory.references = json.loads(p.read_text(encoding="utf-8"))
        return memory
