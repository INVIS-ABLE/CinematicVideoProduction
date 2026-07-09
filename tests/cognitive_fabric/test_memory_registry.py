"""Identity matrix, world memory, memory DB, registry, state resume."""
import torch

from wan.cognitive_fabric.fabric_registry import FabricRegistry
from wan.cognitive_fabric.fabric_state import FabricState
from wan.cognitive_fabric.memory.global_identity_matrix import (
    GlobalIdentityMatrix,
)
from wan.cognitive_fabric.memory.memory_db import MemoryDB
from wan.cognitive_fabric.memory.world_memory import WorldMemory


def test_identity_matrix_roundtrip(tmp_path):
    matrix = GlobalIdentityMatrix(embed_dim=32)
    emb = torch.randn(32)
    matrix.register_character("char_001", "the hero",
                              wardrobe_description="soaked black coat",
                              embedding=emb)
    matrix.register_object("obj_001", "the sword", colour="silver",
                           damage_state="notched")
    matrix.save(str(tmp_path))
    restored = GlobalIdentityMatrix.load(str(tmp_path))
    assert restored.characters["char_001"]["wardrobe_description"] == \
        "soaked black coat"
    assert restored.objects["obj_001"]["damage_state"] == "notched"
    assert torch.equal(restored.embeddings["char_001"], emb)


def test_identity_drift_detection():
    matrix = GlobalIdentityMatrix(embed_dim=16)
    emb = torch.ones(16)
    matrix.register_character("c", "x", embedding=emb)
    assert matrix.compare_identity_drift("c", emb) > 0.99
    assert matrix.compare_identity_drift("c", -emb) < -0.99
    assert matrix.compare_identity_drift("unknown", emb) == 1.0


def test_identity_tokens_and_clauses():
    matrix = GlobalIdentityMatrix(embed_dim=16)
    matrix.register_character("c1", "the hero", face_description="scarred jaw",
                              embedding=torch.randn(16))
    matrix.register_object("o1", "helmet", embedding=torch.randn(16))
    tokens = matrix.get_identity_tokens_for_shot(["c1"], ["o1"])
    assert tokens.shape == (2, 16)
    clauses = matrix.identity_prompt_clauses(["c1"])
    assert any("scarred jaw" in c for c in clauses)
    assert any("do not change" in c for c in clauses)


def test_world_memory_drift(tmp_path):
    worlds = WorldMemory()
    worlds.create_world("w1", location_type="urban cityscape",
                        time_of_day="night", weather="rain")
    issues = worlds.detect_world_drift(
        "w1", {"time_of_day": "day", "weather": "rain"})
    assert len(issues) == 1 and issues[0]["field"] == "time_of_day"
    worlds.save(str(tmp_path))
    restored = WorldMemory.load(str(tmp_path))
    assert restored.worlds["w1"].weather == "rain"
    clauses = restored.get_world_conditioning("w1")
    assert any("night" in c for c in clauses)


def test_memory_db(tmp_path):
    db = MemoryDB(str(tmp_path / "m.db"))
    db.upsert_shot("s1", "scene_001", {"prompt": "p"})
    db.upsert_chunk("s1_c00", "s1", "generating", {})
    db.upsert_chunk("s1_c00", "s1", "approved", {}, attempt=1, output_path="x")
    assert db.chunk_status("s1_c00") == "approved"
    db.add_quality_report("s1_c00", 0.8, {"flicker": 0.9})
    db.add_repair("s1_c00", 0, "adjust_seed", "test")
    assert db.repair_history("s1_c00")[0]["action"] == "adjust_seed"
    db.close()


def test_fabric_state_resume(tmp_path):
    state = FabricState("proj", str(tmp_path / "proj"))
    state.ensure_project_tree()
    state.register_chunk("c1", "s1", "scene_001")
    state.update_chunk("c1", status="approved", output_path="c1.mp4")
    state.register_chunk("c2", "s1", "scene_001")
    state.save_checkpoint()
    resumed = FabricState.resume_or_create("proj", str(tmp_path / "proj"))
    assert resumed.ledger["c1"].status == "approved"
    assert resumed.pending_chunks() == ["c2"]
    assert "resumed_at" in resumed.run_meta


def test_registry_health_check_never_crashes():
    class Broken:
        def health_check(self):
            raise RuntimeError("boom")

    registry = FabricRegistry()
    registry.register("broken", Broken(), "brain")
    registry.register("plain", object(), "memory")
    report = registry.health_check_all()
    assert report["broken"]["ok"] is False
    assert report["plain"]["ok"] is True
