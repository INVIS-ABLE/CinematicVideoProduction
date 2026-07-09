"""Local model registry (spec §20): every base model, adapter, validator and
encoder the engine can load — with license policy enforcement and integrity
hashes. Backed by a JSON file next to the models directory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .allowed_model_sources import check_source

MODEL_TYPES = {
    "video_base", "video_adapter", "lora", "style_adapter", "anime_adapter",
    "character_adapter", "object_adapter", "reward_model", "quality_model",
    "motion_validator", "reference_encoder", "audio_encoder", "speech_model",
    "upscaler", "frame_interpolator", "llm_planner", "vlm", "object_detector",
    "pose_detector",
}


class LocalModelRegistry:
    def __init__(self, registry_path: str = "models/model_registry.json"):
        self.path = Path(registry_path)
        self.models: Dict[str, Dict[str, Any]] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.models = {m["id"]: m for m in payload.get("models", [])}

    def register(self, *, model_id: str, model_type: str, path: str,
                 license: str, precision: str = "bf16",
                 device_policy: str = "gpu", enabled: bool = True
                 ) -> Dict[str, Any]:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unknown model type: {model_type}")
        entry = {"id": model_id, "type": model_type, "path": path,
                 "license": license, "precision": precision,
                 "device_policy": device_policy, "enabled": enabled}
        policy = check_source(entry)
        if not policy["allowed"]:
            raise PermissionError(policy["note"])
        entry["policy"] = policy
        self.models[model_id] = entry
        return entry

    def verify_files(self, model_id: str) -> Dict[str, Any]:
        entry = self.models[model_id]
        root = Path(entry["path"])
        if not root.exists():
            return {"id": model_id, "present": False, "files": 0}
        files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() \
            else [root]
        digest = hashlib.sha256()
        total = 0
        for f in sorted(files):
            digest.update(f.name.encode())
            digest.update(str(f.stat().st_size).encode())
            total += 1
        return {"id": model_id, "present": True, "files": total,
                "manifest_hash": digest.hexdigest()[:16]}

    def enabled_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.models.values()
                if m["type"] == model_type and m.get("enabled")]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"models": list(self.models.values())}, indent=2),
            encoding="utf-8")
        return self.path
