# Wan 2.2 Cognitive Fabric Engine — internal engine upgrade package.
#
# The fabric turns Wan 2.2 from an amnesiac short-clip generator into a
# storyboard-aware, memory-carrying cinematic engine. Everything here either
# runs today, is imported by the generation path, or is a tested interface
# stub with a documented upgrade point (see fable_memory/roadmap.md).
#
# Activation is strictly opt-in via `--cognitive_fabric true` / the
# `cognitive-*` tasks in generate.py. When disabled, original Wan behaviour
# is untouched.

__version__ = "0.1.0"

from .fabric_config import FabricConfig, load_config
from .fabric_types import (
    ChunkSpec,
    CognitiveConditioningPack,
    CognitiveMoERoutingState,
    PhysicsExpectation,
    QualityReport,
    ReferenceToken,
    ShotSpec,
)

__all__ = [
    "__version__",
    "FabricConfig",
    "load_config",
    "ChunkSpec",
    "CognitiveConditioningPack",
    "CognitiveMoERoutingState",
    "PhysicsExpectation",
    "QualityReport",
    "ReferenceToken",
    "ShotSpec",
]
