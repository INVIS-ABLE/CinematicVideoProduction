"""Physics guidance (spec §12): builds PhysicsExpectation objects from shot
plans and converts them to prompt clauses today; the attention-bias tensor
builder is the tensor-shape-safe Phase 4 hook."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..fabric_types import CognitiveConditioningPack, PhysicsExpectation

_ACTION_CONTACTS = {
    "walk": ["feet on ground", "weight shift per step"],
    "run": ["feet strike ground", "forward lean"],
    "jump": ["crouch, launch, airborne arc, landing impact"],
    "fall": ["accelerating descent", "impact deformation"],
    "grab": ["hand wraps object", "fingers occlude grip"],
    "hold": ["continuous hand-object contact"],
    "drive": ["wheels rotate with motion", "suspension response"],
    "fight": ["impact contact", "reactive recoil"],
    "splash": ["water displaced on contact"],
}

_MATERIAL_BEHAVIOURS = {
    "rain": "falls vertically, splashes on impact, wets surfaces",
    "water": "flows downhill, reflects light, ripples from contact",
    "smoke": "rises and disperses with air flow",
    "fire": "flickers upward, emits warm light",
    "cloth": "drapes with gravity, moves with wind and body",
    "hair": "follows head motion with inertia",
    "glass": "transparent, reflects, shatters from impact",
    "mud": "deforms underfoot, sticks",
    "snow": "accumulates, compresses underfoot",
    "dust": "billows from disturbance, settles slowly",
}


def build_physics_expectation(prompt: str,
                              motion_plan: Optional[Dict[str, Any]] = None
                              ) -> PhysicsExpectation:
    text = prompt.lower()
    expectation = PhysicsExpectation()
    for action, contacts in _ACTION_CONTACTS.items():
        if action in text:
            expectation.expected_contacts.extend(contacts)
    for material, behaviour in _MATERIAL_BEHAVIOURS.items():
        if material in text:
            expectation.material_behaviours[material] = behaviour
    if motion_plan:
        path = motion_plan.get("subject_path")
        if path:
            expectation.expected_motion_paths.append(str(path))
    expectation.forbidden_intersections = [
        "characters passing through solid objects",
        "objects intersecting each other",
        "feet sinking below the ground plane",
    ]
    return expectation


def inject_physics_guidance(pack: CognitiveConditioningPack,
                            expectation: PhysicsExpectation
                            ) -> CognitiveConditioningPack:
    pack.prompt_clauses.extend(expectation.to_prompt_clauses())
    pack.negative_clauses.extend([
        "floating objects", "feet sliding", "objects clipping through bodies",
        "shadows pointing the wrong way", "broken hand-object contact",
    ])
    pack.shot_metadata["physics_expectation"] = expectation.__dict__
    return pack


def build_attention_bias_stub(seq_len: int,
                              constraint_pairs: Optional[list] = None
                              ) -> torch.Tensor:
    """Phase 4 hook: additive [L, L] self-attention bias encoding spatial
    constraints. Phase 1 returns zeros of the correct shape/dtype so the
    wiring contract is testable before the real constraint compiler lands."""
    bias = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    return bias
