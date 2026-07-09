"""Attention dispatcher contract tests (refactors R-002/R-004) plus the
Phase-4 context-injection hook contract."""
import torch
import pytest

from wan.modules.attention import attention
from wan.cognitive_fabric.fabric_hooks import (
    attention_backend_report,
    build_context_injection,
    extract_terminal_frames,
)


def test_sdpa_fallback_preserves_shape_dtype_device():
    q = torch.randn(2, 16, 4, 8)  # [B, L, N, d]
    k = torch.randn(2, 16, 4, 8)
    v = torch.randn(2, 16, 4, 8)
    out = attention(q, k, v)
    assert out.shape == (2, 16, 4, 8)
    assert out.dtype == q.dtype          # R-004: dtype in == dtype out
    assert out.device == q.device


def test_sdpa_fallback_cross_attention_shapes():
    """Cross-attn: kv length differs from q length (text context 512)."""
    q = torch.randn(1, 48, 4, 8)
    k = torch.randn(1, 12, 4, 8)
    v = torch.randn(1, 12, 4, 8)
    out = attention(q, k, v)
    assert out.shape == (1, 48, 4, 8)


def test_sdpa_matches_manual_sdpa():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    v = torch.randn(1, 8, 2, 4)
    out = attention(q, k, v, dtype=torch.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    assert torch.allclose(out, ref, atol=1e-5)


def test_backend_report():
    report = attention_backend_report()
    assert report["selected_backend"] in (
        "flash_attn_3", "flash_attn_2", "torch_sdpa_fallback")


def test_context_injection_contract():
    ctx = torch.randn(2, 512, 64)
    extra = torch.randn(5, 64)
    out = build_context_injection(ctx, extra)
    assert out.shape == (2, 517, 64)
    assert torch.equal(out[:, :512], ctx)
    assert build_context_injection(ctx, None) is ctx
    with pytest.raises(ValueError):
        build_context_injection(ctx, torch.randn(5, 32))  # dim mismatch


def test_terminal_frame_extraction():
    video = torch.randn(3, 20, 16, 16)
    tail = extract_terminal_frames(video, count=8)
    assert tail.shape == (3, 8, 16, 16)
    assert torch.equal(tail, video[:, -8:])
