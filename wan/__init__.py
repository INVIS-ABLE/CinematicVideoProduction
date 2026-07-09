# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Cognitive Fabric refactor R-001 (see fable_memory/refactor_log.md):
# WanS2V and WanAnimate pull heavy optional dependencies (librosa, decord,
# cv2, peft) that live in requirements_s2v.txt / requirements_animate.txt,
# not in the base requirements. Importing them unconditionally made
# `import wan` crash on a base install. They are now exported lazily via
# PEP 562 module __getattr__ — public API is unchanged, and environments
# with the extras installed behave exactly as before.
from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V

_LAZY_PIPELINES = {
    "WanS2V": ("wan.speech2video", "WanS2V"),
    "WanAnimate": ("wan.animate", "WanAnimate"),
}

__all__ = [
    "configs",
    "distributed",
    "modules",
    "WanI2V",
    "WanT2V",
    "WanTI2V",
    "WanS2V",
    "WanAnimate",
    "cognitive_fabric",
]


def __getattr__(name):
    if name in _LAZY_PIPELINES:
        import importlib

        module_name, attr = _LAZY_PIPELINES[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError(
                f"{name} requires optional dependencies "
                f"(see requirements_s2v.txt / requirements_animate.txt): {e}"
            ) from e
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent lookups
        return value
    if name == "cognitive_fabric":
        import importlib

        module = importlib.import_module("wan.cognitive_fabric")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'wan' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_PIPELINES.keys()))
