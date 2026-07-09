"""Component registry: every brain / memory / pipeline organ registers here
so the runtime can health-check, list, and replace them individually."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RegisteredComponent:
    name: str
    kind: str  # brain | memory | pipeline | conditioning | validator | install
    component: Any
    required: bool = False


class FabricRegistry:
    def __init__(self):
        self._components: Dict[str, RegisteredComponent] = {}

    def register(self, name: str, component: Any, kind: str,
                 required: bool = False) -> None:
        self._components[name] = RegisteredComponent(
            name=name, kind=kind, component=component, required=required)

    def get(self, name: str) -> Optional[Any]:
        entry = self._components.get(name)
        return entry.component if entry else None

    def require(self, name: str) -> Any:
        component = self.get(name)
        if component is None:
            raise KeyError(f"required fabric component missing: {name}")
        return component

    def by_kind(self, kind: str) -> List[RegisteredComponent]:
        return [c for c in self._components.values() if c.kind == kind]

    def names(self) -> List[str]:
        return sorted(self._components.keys())

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        report: Dict[str, Dict[str, Any]] = {}
        for name, entry in sorted(self._components.items()):
            check = getattr(entry.component, "health_check", None)
            try:
                detail = check() if callable(check) else {"status": "ok",
                                                          "note": "no check"}
                ok = detail.get("status", "ok") == "ok"
            except Exception as e:  # a failing organ must never crash the fabric
                detail, ok = {"status": "error", "note": str(e)}, False
            report[name] = {
                "kind": entry.kind,
                "required": entry.required,
                "ok": ok,
                **detail,
            }
        return report
