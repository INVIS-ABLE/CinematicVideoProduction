"""Collision guidance (spec §12): no-penetration / contact constraints.
Phase 1 emits constraint records + prompt clauses; mask building is the
tensor-shape-safe hook consumed by physics_guidance.build_attention_bias_stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CollisionConstraint:
    subject: str
    other: str
    relation: str  # no_penetration | contact | occlusion_order
    notes: str = ""


@dataclass
class CollisionPlan:
    constraints: List[CollisionConstraint] = field(default_factory=list)

    def add_no_penetration(self, subject: str, other: str, notes: str = "") -> None:
        self.constraints.append(CollisionConstraint(subject, other,
                                                    "no_penetration", notes))

    def add_contact(self, subject: str, other: str, notes: str = "") -> None:
        self.constraints.append(CollisionConstraint(subject, other,
                                                    "contact", notes))

    def prompt_clauses(self) -> List[str]:
        clauses = []
        for c in self.constraints:
            if c.relation == "contact":
                clauses.append(f"{c.subject} makes solid contact with {c.other}")
            elif c.relation == "no_penetration":
                clauses.append(f"{c.subject} never passes through {c.other}")
        return clauses


def build_collision_plan(character_ids: List[str],
                         object_ids: List[str]) -> CollisionPlan:
    plan = CollisionPlan()
    for cid in character_ids:
        plan.add_no_penetration(cid, "walls and solid props")
        plan.add_contact(cid, "the ground plane")
        for oid in object_ids:
            plan.add_no_penetration(cid, oid)
    return plan
