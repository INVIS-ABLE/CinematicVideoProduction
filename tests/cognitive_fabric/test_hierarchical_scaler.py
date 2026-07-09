"""Hierarchical scaler: tiled processing, seam-free blending, stage sizes,
finishing ops, interpolation."""
import torch

from wan.cognitive_fabric.brains.upscale_brain import UpscaleBrain
from wan.cognitive_fabric.pipeline.frame_interpolator import FrameInterpolator
from wan.cognitive_fabric.pipeline.hierarchical_scaler import (
    HierarchicalScaler,
    tiled_enhance,
)


def test_tiled_identity_has_no_seams():
    """Feathered tile blending of an identity op must reproduce the input —
    the strongest possible seam test."""
    torch.manual_seed(0)
    frame = torch.rand(3, 100, 140) * 2 - 1
    out = tiled_enhance(frame, lambda t: t, scale=1, tile=48, overlap=16)
    assert out.shape == frame.shape
    assert torch.allclose(out, frame, atol=1e-5)


def test_tiled_upscale_matches_direct_within_tolerance():
    torch.manual_seed(1)
    frame = torch.rand(3, 64, 96) * 2 - 1
    direct = torch.nn.functional.interpolate(
        frame.unsqueeze(0), scale_factor=2, mode="bicubic",
        align_corners=False).squeeze(0).clamp(-1, 1)
    tiled = tiled_enhance(
        frame,
        lambda t: torch.nn.functional.interpolate(
            t.unsqueeze(0), scale_factor=2, mode="bicubic",
            align_corners=False).squeeze(0).clamp(-1, 1),
        scale=2, tile=48, overlap=16)
    assert tiled.shape == direct.shape
    # interiors match; only feathered borders may differ slightly
    assert (tiled - direct).abs().mean() < 0.02


def test_stage_output_sizes():
    scaler = HierarchicalScaler(tile=64, overlap=16)
    video = torch.rand(3, 2, 90, 160) * 2 - 1  # tiny 16:9-ish source
    hd = scaler.upscale_to_1080p(video)
    assert hd.shape == (3, 2, 1080, 1920)
    uhd = scaler.upscale_to_4k(video)
    assert uhd.shape == (3, 2, 2160, 3840)


def test_temporal_denoise_and_sharpen_preserve_shape():
    scaler = HierarchicalScaler()
    video = torch.rand(3, 6, 32, 32) * 2 - 1
    assert scaler.denoise_temporal(video).shape == video.shape
    assert scaler.sharpen_temporal(video).shape == video.shape
    assert scaler.apply_colour_finish(video, gamma=1.1,
                                      saturation=1.2).shape == video.shape


def test_interpolator_frame_count():
    video = torch.rand(3, 5, 16, 16)
    out = FrameInterpolator().interpolate(video, factor=2)
    assert out.shape[1] == 9  # (5-1)*2+1
    assert torch.equal(out[:, 0], video[:, 0])
    assert torch.equal(out[:, -1], video[:, -1])


def test_upscale_brain_stage_planning():
    brain = UpscaleBrain()
    assert brain.plan_stages("720p", "4k") == ["hd", "qhd", "uhd"]
    assert brain.plan_stages("720p", "1080p") == ["hd"]
    assert brain.plan_stages("720p", "720p") == []


def test_custom_backend_registration():
    scaler = HierarchicalScaler()
    calls = []

    def fake_backend(tile):
        calls.append(1)
        return torch.nn.functional.interpolate(
            tile.unsqueeze(0), scale_factor=1, mode="nearest").squeeze(0)

    scaler.register_backend("fake", fake_backend)
    video = torch.rand(3, 1, 32, 32)
    scaler._enhance_video(video, (64, 36), backend="fake")
    assert calls  # custom backend was actually used
