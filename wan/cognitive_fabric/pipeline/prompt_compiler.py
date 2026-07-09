"""Prompt compiler: assembles the CognitiveConditioningPack for one chunk
from every fabric source, then compiles final prompt + negative strings.

Order of assembly (user idea always first, continuity always attached):
user prompt → identity → objects → world → camera → lighting → physics →
style → continuity clauses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..brains.prompt_brain import PromptBrain
from ..conditioning.camera_guidance import inject_camera_guidance
from ..conditioning.collision_guidance import build_collision_plan
from ..conditioning.identity_token_injector import inject_identity_tokens
from ..conditioning.lighting_guidance import inject_lighting_guidance
from ..conditioning.object_token_injector import inject_object_tokens
from ..conditioning.physics_guidance import (
    build_physics_expectation,
    inject_physics_guidance,
)
from ..conditioning.reference_token_stack import ReferenceTokenStack
from ..conditioning.storyboard_guidance import build_routing_state
from ..conditioning.style_guidance import inject_style_guidance
from ..conditioning.world_token_injector import inject_world_tokens
from ..fabric_types import ChunkSpec, CognitiveConditioningPack
from ..memory.global_identity_matrix import GlobalIdentityMatrix
from ..memory.object_memory import ObjectMemory
from ..memory.style_memory import StyleMemory
from ..memory.world_memory import WorldMemory


class PromptCompiler:
    def __init__(self, *, matrix: GlobalIdentityMatrix, worlds: WorldMemory,
                 objects: ObjectMemory, style: StyleMemory,
                 reference_stack: Optional[ReferenceTokenStack] = None,
                 prompt_brain: Optional[PromptBrain] = None):
        self.matrix = matrix
        self.worlds = worlds
        self.objects = objects
        self.style = style
        self.reference_stack = reference_stack
        self.prompt_brain = prompt_brain or PromptBrain()

    def compile_chunk(self, chunk: ChunkSpec, shot: Dict[str, Any],
                      continuity_clauses: Optional[List[str]] = None
                      ) -> Dict[str, Any]:
        pack = CognitiveConditioningPack()
        pack.shot_metadata = {
            "shot_id": shot["shot_id"],
            "shot_type": shot.get("shot_type", "medium"),
            "character_ids": shot.get("character_ids", []),
            "object_ids": shot.get("object_ids", []),
            "world_id": shot.get("world_id"),
            "camera_plan": shot.get("camera_plan", {}),
            "lighting_plan": shot.get("lighting_plan", {}),
            "physics_expectation": shot.get("physics_expectation", {}),
        }
        pack.scene_metadata = {"scene_id": shot.get("scene_id")}

        inject_identity_tokens(pack, self.matrix,
                               shot.get("character_ids", []),
                               shot.get("object_ids", []))
        inject_object_tokens(pack, self.objects, shot.get("object_ids", []))
        inject_world_tokens(pack, self.worlds, shot.get("world_id"))
        inject_camera_guidance(pack, shot.get("camera_plan", {}))
        inject_lighting_guidance(pack, shot.get("lighting_plan", {}))

        expectation = build_physics_expectation(chunk.prompt,
                                                shot.get("motion_plan"))
        collision = build_collision_plan(shot.get("character_ids", []),
                                         shot.get("object_ids", []))
        expectation.forbidden_intersections.extend(
            c.notes for c in collision.constraints if c.notes)
        pack.prompt_clauses.extend(collision.prompt_clauses()[:4])
        inject_physics_guidance(pack, expectation)

        inject_style_guidance(pack, self.style)

        if continuity_clauses:
            pack.prompt_clauses.extend(continuity_clauses)

        if self.reference_stack is not None:
            self.reference_stack.to_conditioning_pack(
                scene_id=shot.get("scene_id"), shot_id=shot["shot_id"],
                base=pack)

        routing = build_routing_state(pack.shot_metadata)
        pack.timestep_metadata["moe_routing"] = routing.__dict__.copy()
        pack.timestep_metadata["moe_routing"].pop("repair_focus_mask", None)

        pack.validate()
        prompt = self.prompt_brain.compile_prompt(chunk.prompt, pack)
        negative = self.prompt_brain.compile_negative(chunk.negative_prompt, pack)
        return {"prompt": prompt, "negative_prompt": negative, "pack": pack,
                "routing": routing}
