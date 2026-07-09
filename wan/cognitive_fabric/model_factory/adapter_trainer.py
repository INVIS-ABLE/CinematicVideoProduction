"""Adapter trainer — planned hook (Phase 8). Interface + dataset contract
pinned now; training loops land with the model factory phase."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AdapterTrainingSpec:
    adapter_id: str
    adapter_type: str  # style|character|object|world|motion
    base_model_id: str
    dataset_paths: List[str] = field(default_factory=list)
    rank: int = 32
    steps: int = 2000
    learning_rate: float = 1e-4
    licensed_sources_only: bool = True


class AdapterTrainer:
    def prepare(self, spec: AdapterTrainingSpec) -> Dict[str, Any]:
        if not spec.licensed_sources_only:
            raise PermissionError(
                "adapter training requires licensed_sources_only=True")
        return {"spec": spec.__dict__, "status": "prepared",
                "note": "training loop lands in Phase 8"}

    def train(self, spec: AdapterTrainingSpec) -> Dict[str, Any]:
        raise NotImplementedError("adapter training lands in Phase 8")
