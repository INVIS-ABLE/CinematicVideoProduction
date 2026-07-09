"""Global Identity Matrix — the heart of character/object continuation
(spec §7). Profiles are JSON; embeddings are torch tensors persisted next to
the JSON as .pt. If a character returns 30 minutes later, its profile and
embedding reload and re-condition generation.

Phase 1 embeddings come from the deterministic reference encoder (see
conditioning/reference_token_stack.py); stronger encoders slot in through
the same `embedding` fields without schema change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a @ b) / denom)


class GlobalIdentityMatrix:
    def __init__(self, embed_dim: int = 256):
        self.embed_dim = embed_dim
        self.characters: Dict[str, Dict[str, Any]] = {}
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.world_anchors: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, torch.Tensor] = {}  # key = profile id

    # ---- registration ------------------------------------------------------

    def register_character(self, character_id: str, name: str, *,
                           role: str = "", face_description: str = "",
                           body_description: str = "",
                           hair_description: str = "",
                           skin_description: str = "",
                           wardrobe_description: str = "",
                           colour_palette: str = "",
                           anime_style: str = "",
                           approved_reference_frames: Optional[List[str]] = None,
                           continuity_rules: Optional[List[str]] = None,
                           allowed_variation: str = "expression, pose",
                           forbidden_changes: str = "face, wardrobe, age, body type",
                           embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        profile = {
            "character_id": character_id,
            "name": name,
            "role": role,
            "face_description": face_description,
            "body_description": body_description,
            "hair_description": hair_description,
            "skin_description": skin_description,
            "wardrobe_description": wardrobe_description,
            "colour_palette": colour_palette,
            "anime_style_embedding": anime_style,
            "approved_reference_frames": approved_reference_frames or [],
            "rejected_drift_frames": [],
            "continuity_rules": continuity_rules or [],
            "last_seen_scene": None,
            "last_seen_shot": None,
            "allowed_variation": allowed_variation,
            "forbidden_changes": forbidden_changes,
            "registered_at": time.time(),
        }
        self.characters[character_id] = profile
        if embedding is not None:
            self.embeddings[character_id] = embedding.detach().clone()
        return profile

    def register_object(self, object_id: str, name: str, *,
                        object_type: str = "prop", material: str = "",
                        colour: str = "", shape: str = "", size: str = "",
                        damage_state: str = "intact",
                        owner_character_id: Optional[str] = None,
                        continuity_rules: Optional[List[str]] = None,
                        allowed_variation: str = "viewing angle",
                        forbidden_changes: str = "colour, shape, damage state, logo",
                        embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        profile = {
            "object_id": object_id,
            "name": name,
            "type": object_type,
            "material": material,
            "colour": colour,
            "shape": shape,
            "size": size,
            "damage_state": damage_state,
            "owner_character_id": owner_character_id,
            "approved_frames": [],
            "continuity_rules": continuity_rules or [],
            "last_seen_scene": None,
            "last_seen_shot": None,
            "allowed_variation": allowed_variation,
            "forbidden_changes": forbidden_changes,
            "registered_at": time.time(),
        }
        self.objects[object_id] = profile
        if embedding is not None:
            self.embeddings[object_id] = embedding.detach().clone()
        return profile

    def register_world_anchor(self, anchor_id: str, description: str, *,
                              embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        profile = {"anchor_id": anchor_id, "description": description,
                   "registered_at": time.time()}
        self.world_anchors[anchor_id] = profile
        if embedding is not None:
            self.embeddings[anchor_id] = embedding.detach().clone()
        return profile

    # ---- updates -------------------------------------------------------------

    def update_from_approved_frame(self, profile_id: str, frame_path: str,
                                   scene_id: Optional[str] = None,
                                   shot_id: Optional[str] = None,
                                   embedding: Optional[torch.Tensor] = None) -> None:
        if profile_id in self.characters:
            profile = self.characters[profile_id]
            profile["approved_reference_frames"].append(frame_path)
        elif profile_id in self.objects:
            profile = self.objects[profile_id]
            profile["approved_frames"].append(frame_path)
        else:
            raise KeyError(f"unknown identity profile: {profile_id}")
        profile["last_seen_scene"] = scene_id
        profile["last_seen_shot"] = shot_id
        if embedding is not None:
            # exponential moving average keeps identity stable while adapting
            prev = self.embeddings.get(profile_id)
            self.embeddings[profile_id] = (
                embedding.detach().clone() if prev is None
                else 0.8 * prev + 0.2 * embedding.detach())

    # ---- retrieval -------------------------------------------------------------

    def get_identity_tokens_for_shot(self, character_ids: List[str],
                                     object_ids: List[str]) -> Optional[torch.Tensor]:
        rows = [self.embeddings[i]
                for i in list(character_ids) + list(object_ids)
                if i in self.embeddings]
        if not rows:
            return None
        return torch.stack([r.flatten()[:self.embed_dim] for r in rows])

    def identity_prompt_clauses(self, character_ids: List[str]) -> List[str]:
        clauses = []
        for cid in character_ids:
            p = self.characters.get(cid)
            if not p:
                continue
            bits = [p["name"]]
            for key in ("face_description", "hair_description",
                        "wardrobe_description", "colour_palette"):
                if p.get(key):
                    bits.append(p[key])
            clauses.append("consistent character: " + ", ".join(bits))
            if p.get("forbidden_changes"):
                clauses.append(
                    f"do not change {p['name']}'s {p['forbidden_changes']}")
        return clauses

    def compare_identity_drift(self, profile_id: str,
                               candidate_embedding: torch.Tensor) -> float:
        """Return similarity in [−1, 1]; below the configured threshold the
        repair loop treats the chunk as identity drift."""
        reference = self.embeddings.get(profile_id)
        if reference is None:
            return 1.0  # nothing to compare against — no drift signal
        return _cosine(reference, candidate_embedding)

    # ---- persistence ------------------------------------------------------------

    def save(self, directory: str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "identity_matrix.json").write_text(json.dumps({
            "embed_dim": self.embed_dim,
            "characters": self.characters,
            "objects": self.objects,
            "world_anchors": self.world_anchors,
        }, indent=2), encoding="utf-8")
        if self.embeddings:
            torch.save(self.embeddings, d / "identity_embeddings.pt")
        return d / "identity_matrix.json"

    @classmethod
    def load(cls, directory: str) -> "GlobalIdentityMatrix":
        d = Path(directory)
        payload = json.loads((d / "identity_matrix.json").read_text(encoding="utf-8"))
        matrix = cls(embed_dim=payload.get("embed_dim", 256))
        matrix.characters = payload.get("characters", {})
        matrix.objects = payload.get("objects", {})
        matrix.world_anchors = payload.get("world_anchors", {})
        pt = d / "identity_embeddings.pt"
        if pt.is_file():
            matrix.embeddings = torch.load(pt, weights_only=True)
        return matrix
