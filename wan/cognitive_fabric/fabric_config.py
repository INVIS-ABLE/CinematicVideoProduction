"""Cognitive fabric configuration.

The full default lives here in code so the engine never needs a config file
to boot; YAML/JSON files under configs/ override it (deep merge). YAML is
optional — JSON always works.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "cognitive_fabric": {
        "enabled": True,
        "mode": "cinematic",
        "target_output": "film",
        "storyboard_mode": True,
        "longform_enabled": True,
        "max_minutes": 60,
        "base_resolution": "720p",
        "refinement_resolution": "1080p",
        "qhd_resolution": "1440p",
        "final_resolution": "4k",
        "frame_rate": 24,
        "optional_master_frame_rate": 60,
        "chunk_seconds": 5,
        "overlap_frames": 8,
        "temporal_cache_enabled": True,
        "identity_memory_enabled": True,
        "object_memory_enabled": True,
        "world_memory_enabled": True,
        "physics_guidance_enabled": True,
        "collision_guidance_enabled": True,
        "reference_token_limit": 50,
        "repair_loop_enabled": True,
        "max_regenerations_per_chunk": 3,
        "anime_mode_enabled": False,
        "model_factory_enabled": True,
        "local_only": True,
    },
    "temporal_memory": {
        "overlap_frames": 8,
        "max_cache_chunks": 12,
        "decay_rate": 0.82,
        "preserve_identity_tokens": True,
        "preserve_world_tokens": True,
        "scene_boundary_decay": 0.35,
        "hard_scene_cut_resets_motion": True,
        "soft_transition_preserves_motion": True,
        "terminal_frames": 8,
    },
    "repair": {
        "max_attempts_per_chunk": 3,
        "identity_threshold": 0.82,
        "world_threshold": 0.75,
        "physics_threshold": 0.70,
        "anatomy_threshold": 0.78,
        "cinematic_threshold": 0.72,
    },
    "resolution_pipeline": {
        "base": {"name": "720p", "width": 1280, "height": 720},
        "hd": {"name": "1080p", "width": 1920, "height": 1080},
        "qhd": {"name": "1440p", "width": 2560, "height": 1440},
        "uhd": {"name": "4k", "width": 3840, "height": 2160},
    },
    "hardware_modes": {
        "low_vram": {
            "base_resolution": "480p", "chunk_seconds": 3,
            "reference_token_limit": 12, "offload": True, "dtype": "fp16",
            "min_vram_gb": 0,
        },
        "creator": {
            "base_resolution": "720p", "chunk_seconds": 5,
            "reference_token_limit": 25, "offload": True, "dtype": "bf16",
            "min_vram_gb": 12,
        },
        "studio_4090": {
            "base_resolution": "720p", "chunk_seconds": 5,
            "reference_token_limit": 50, "offload": "smart", "dtype": "bf16",
            "min_vram_gb": 22,
        },
        "multi_gpu": {
            "base_resolution": "720p", "chunk_seconds": 8,
            "reference_token_limit": 50, "distributed": True, "dtype": "bf16",
            "min_vram_gb": 40,
        },
    },
    "engine": {
        # preferred wan task per fabric mode; ti2v-5B is default because its
        # i2v latent clamp is the continuity mechanism.
        "preferred_task": "ti2v-5B",
        "sample_solver": "unipc",
        "reference_embed_dim": 256,
    },
}

# resolutions the base generator actually supports natively; anything above
# is reached through the hierarchical scaler, never claimed as native.
NATIVE_GENERATION_SIZES = {
    "480p": (832, 480),
    "720p": (1280, 704),  # TI2V-5B native grid (multiples of 32)
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class FabricConfig:
    """Dict-backed config with dotted-path access and validation."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = _deep_merge(DEFAULT_CONFIG, data or {})
        self.validate()

    @classmethod
    def load(cls, path: Optional[str] = None,
             overrides: Optional[Dict[str, Any]] = None) -> "FabricConfig":
        data: Dict[str, Any] = {}
        if path:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f"fabric config not found: {path}")
            text = p.read_text(encoding="utf-8")
            if p.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                except ImportError as e:
                    raise ImportError(
                        "PyYAML is required for YAML configs; "
                        "use a .json config or `pip install pyyaml`") from e
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
        if overrides:
            data = _deep_merge(data, overrides)
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def validate(self) -> None:
        cf = self.data["cognitive_fabric"]
        if cf["chunk_seconds"] <= 0:
            raise ValueError("chunk_seconds must be > 0")
        if cf["max_regenerations_per_chunk"] < 0:
            raise ValueError("max_regenerations_per_chunk must be >= 0")
        if cf["reference_token_limit"] < 1:
            raise ValueError("reference_token_limit must be >= 1")
        for stage, spec in self.data["resolution_pipeline"].items():
            if spec["width"] % 2 or spec["height"] % 2:
                raise ValueError(f"resolution_pipeline.{stage} must be even")
        for name in ("base_resolution", "final_resolution"):
            if not isinstance(cf[name], str):
                raise ValueError(f"{name} must be a string like '720p'")

    def resolution(self, name: str) -> tuple:
        """Resolve a name like '720p'/'4k' to (width, height)."""
        aliases = {"4k": "uhd", "2160p": "uhd", "1440p": "qhd",
                   "1080p": "hd", "720p": "base", "480p": "base"}
        if name == "480p":
            return NATIVE_GENERATION_SIZES["480p"]
        stage = aliases.get(name, name)
        spec = self.data["resolution_pipeline"].get(stage)
        if spec is None:
            raise ValueError(f"unknown resolution: {name}")
        return (spec["width"], spec["height"])

    def native_generation_size(self, name: str) -> tuple:
        """Size actually handed to Wan (never above native support)."""
        return NATIVE_GENERATION_SIZES.get(name, NATIVE_GENERATION_SIZES["720p"])

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def load_config(path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> FabricConfig:
    return FabricConfig.load(path, overrides) if path or overrides else FabricConfig()
