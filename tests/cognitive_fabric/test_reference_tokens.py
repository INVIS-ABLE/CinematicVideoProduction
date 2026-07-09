"""Reference token stack: encoding determinism, priority pruning, manifest
honesty, conditioning pack contract."""
import torch

from wan.cognitive_fabric.conditioning.reference_token_stack import (
    DeterministicReferenceEncoder,
    ReferenceTokenStack,
)


def test_encoder_is_deterministic(tmp_path):
    f = tmp_path / "ref.png"
    f.write_bytes(b"fake-image-bytes-1234")
    enc = DeterministicReferenceEncoder(dim=64)
    a = enc(str(f), "image")
    b = enc(str(f), "image")
    assert torch.equal(a, b)
    assert a.shape == (64,)
    # different role → different embedding
    c = enc(str(f), "style")
    assert not torch.equal(a, c)


def test_stack_compiles_weighted_matrix(tmp_path):
    stack = ReferenceTokenStack(token_limit=10, dim=32)
    for i in range(3):
        f = tmp_path / f"r{i}.png"
        f.write_bytes(bytes([i]) * 100)
        stack.add_image_reference(str(f), weight=0.5)
    tokens, manifest = stack.compile_tokens()
    assert tokens.shape == (3, 32)
    assert len(manifest) == 3 and all(m["used"] for m in manifest)


def test_priority_pruning_keeps_identity_refs(tmp_path):
    stack = ReferenceTokenStack(token_limit=3, dim=16)
    char = tmp_path / "char.png"
    char.write_bytes(b"character-sheet")
    stack.add_character_reference(str(char), "char_001")  # priority 9
    for i in range(4):
        f = tmp_path / f"style{i}.png"
        f.write_bytes(bytes([i + 10]) * 50)
        stack.add_style_reference(str(f), priority=2)
    tokens, manifest = stack.compile_tokens()
    assert tokens.shape[0] == 3
    used_roles = [m["role"] for m in manifest if m["used"]]
    assert "character_sheet" in used_roles          # identity ref survived
    pruned = [m for m in manifest if not m["used"]]
    assert len(pruned) == 2                          # nothing silently dropped
    assert all(m["reason"] == "pruned_over_limit" for m in pruned)


def test_scope_filtering(tmp_path):
    stack = ReferenceTokenStack(token_limit=10, dim=16)
    f1 = tmp_path / "a.png"; f1.write_bytes(b"a")
    f2 = tmp_path / "b.png"; f2.write_bytes(b"b")
    stack.add_image_reference(str(f1), shot_scope="shot_001")
    stack.add_image_reference(str(f2))  # global
    tokens, _ = stack.compile_tokens(shot_id="shot_002")
    assert tokens.shape[0] == 1  # shot_001-scoped ref excluded


def test_to_conditioning_pack(tmp_path):
    stack = ReferenceTokenStack(token_limit=10, dim=16)
    f = tmp_path / "w.png"; f.write_bytes(b"world")
    stack.add_world_reference(str(f))
    pack = stack.to_conditioning_pack()
    pack.validate()
    assert pack.reference_tokens.shape == (1, 16)
    assert pack.token_count() == 1
