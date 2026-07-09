"""Style memory — the project's locked visual grammar (style bible)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class StyleMemory:
    def __init__(self, style: Dict[str, Any] = None):
        self.style: Dict[str, Any] = style or {}

    def lock(self, **kwargs: Any) -> None:
        self.style.update(kwargs)

    def prompt_clauses(self) -> List[str]:
        clauses = []
        for key in ("look", "palette", "grain", "lens_character", "grade"):
            if self.style.get(key):
                clauses.append(f"{key.replace('_', ' ')}: {self.style[key]}")
        return clauses

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.style, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str) -> "StyleMemory":
        p = Path(path)
        return cls(json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {})
