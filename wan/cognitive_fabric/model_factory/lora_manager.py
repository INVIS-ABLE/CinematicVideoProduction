"""LoRA manager — planned hook (Phase 8, see fable_memory/roadmap.md).

The base Wan repo only uses peft inside the Animate relighting LoRA; a
general LoRA route for T2V/TI2V requires adapter key-mapping work against
WanModel's module names. This module pins the interface; `load()` raises a
clear NotImplementedError until Phase 8 lands rather than pretending."""
from __future__ import annotations

from typing import Any, Dict, List

from .local_model_registry import LocalModelRegistry


class LoraManager:
    def __init__(self, registry: LocalModelRegistry):
        self.registry = registry

    def available(self) -> List[Dict[str, Any]]:
        return self.registry.enabled_by_type("lora")

    def load(self, model: Any, lora_id: str, strength: float = 1.0) -> Any:
        raise NotImplementedError(
            "LoRA loading for WanModel lands in Phase 8 (model factory); "
            "adapters are registered and policy-checked today via "
            "LocalModelRegistry, not silently ignored.")
