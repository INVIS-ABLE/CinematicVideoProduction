"""Model clone/adaptation manager (spec §20) — policy-gated interface.
'Cloning' means creating local functional equivalents from open/licensed/
user-owned sources only; the policy check is enforced, extraction of
proprietary weights is refused."""
from __future__ import annotations

from typing import Any, Dict

from .allowed_model_sources import check_source
from .local_model_registry import LocalModelRegistry


class ModelCloneManager:
    def __init__(self, registry: LocalModelRegistry):
        self.registry = registry

    def ingest(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        policy = check_source(entry)
        if not policy["allowed"]:
            raise PermissionError(policy["note"])
        return self.registry.register(
            model_id=entry["id"], model_type=entry["type"],
            path=entry["path"], license=entry["license"],
            precision=entry.get("precision", "bf16"),
            device_policy=entry.get("device_policy", "gpu"),
            enabled=entry.get("enabled", True))
