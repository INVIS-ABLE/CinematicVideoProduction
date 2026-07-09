"""style_adapter_factory — planned hook (Phase 8, see fable_memory/roadmap.md). Interface
pinned via AdapterTrainer/AdapterTrainingSpec; raises rather than fakes."""
from __future__ import annotations

from .adapter_trainer import AdapterTrainer, AdapterTrainingSpec

__all__ = ["AdapterTrainer", "AdapterTrainingSpec"]
