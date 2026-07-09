"""Temporal KV cache: capture/retrieve/decay/prune + checkpoint roundtrip."""
import torch

from wan.cognitive_fabric.memory.temporal_kv_cache import (
    TemporalKVCache,
    compress_frames_to_embedding,
)


def _frames(seed=0):
    torch.manual_seed(seed)
    return torch.rand(3, 8, 16, 16) * 2 - 1


def test_capture_and_retrieve():
    cache = TemporalKVCache(max_cache_chunks=4)
    cache.capture("c1", "scene_001", terminal_frames=_frames(1))
    cache.capture("c2", "scene_001", terminal_frames=_frames(2))
    latest = cache.latest()
    assert latest.chunk_id == "c2"
    assert latest.visual_embedding.shape == (256,)
    assert len(cache.retrieve(scene_id="scene_001", last_n=2)) == 2


def test_prune_respects_limit():
    cache = TemporalKVCache(max_cache_chunks=3)
    for i in range(6):
        cache.capture(f"c{i}", "s", terminal_frames=_frames(i))
    assert len(cache) == 3
    assert cache.latest().chunk_id == "c5"


def test_decay_drops_heavy_payload_keeps_identity():
    cache = TemporalKVCache(decay_rate=0.1, preserve_identity_tokens=True)
    cache.capture("c1", "s", terminal_frames=_frames(),
                  identity_tokens=torch.randn(2, 256))
    for _ in range(3):
        cache.decay()
    entry = cache.latest()
    assert entry.terminal_frames is None       # heavy payload released
    assert entry.identity_tokens is not None   # identity preserved


def test_scene_boundary_decay_is_stronger():
    a = TemporalKVCache(decay_rate=0.9, scene_boundary_decay=0.2)
    a.capture("c1", "s", terminal_frames=_frames())
    a.decay(scene_boundary=True)
    assert a.latest().weight == 0.2


def test_continuity_similarity_signal():
    cache = TemporalKVCache()
    frames = _frames(7)
    cache.capture("c1", "s", terminal_frames=frames)
    same = cache.continuity_similarity(frames)
    different = cache.continuity_similarity(-frames)
    assert same > 0.99
    assert different < same


def test_checkpoint_roundtrip(tmp_path):
    cache = TemporalKVCache(max_cache_chunks=5)
    cache.capture("c1", "s", terminal_frames=_frames(3),
                  identity_tokens=torch.randn(1, 256))
    path = tmp_path / "cache.pt"
    cache.save_checkpoint(str(path))
    restored = TemporalKVCache.load_checkpoint(str(path))
    assert len(restored) == 1
    assert restored.latest().chunk_id == "c1"
    assert torch.equal(restored.latest().identity_tokens,
                       cache.latest().identity_tokens)


def test_embedding_shape_contract():
    emb = compress_frames_to_embedding(_frames(), dim=128)
    assert emb.shape == (128,)
