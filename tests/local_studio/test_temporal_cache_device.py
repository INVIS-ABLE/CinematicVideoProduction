import torch

from wan.cognitive_fabric.memory.temporal_kv_cache import TemporalKVCache


def test_cache_retains_frames_and_embeddings_on_cpu():
    frames = torch.rand(3, 4, 16, 16)
    cache = TemporalKVCache()
    entry = cache.capture("chunk_1", "scene_1", terminal_frames=frames)
    assert entry.terminal_frames.device.type == "cpu"
    assert entry.visual_embedding.device.type == "cpu"
    assert cache.continuity_similarity(frames) is not None


def test_cuda_input_is_moved_off_gpu_when_available():
    if not torch.cuda.is_available():
        return
    frames = torch.rand(3, 4, 16, 16, device="cuda")
    entry = TemporalKVCache().capture("chunk_1", "scene_1", terminal_frames=frames)
    assert entry.terminal_frames.device.type == "cpu"
    assert entry.visual_embedding.device.type == "cpu"
