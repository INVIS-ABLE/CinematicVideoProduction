"""Verify the tensor-shape map (fable_memory/wan22_tensor_shape_map.md) by
running the REAL WanModel forward on CPU through the SDPA fallback
(refactors R-002/R-004). This is the regression gate for any future
attention/conditioning change."""
import torch
import pytest

from wan.modules.model import WanModel


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return WanModel(model_type="t2v", patch_size=(1, 2, 2), text_len=16,
                    in_dim=4, dim=64, ffn_dim=128, freq_dim=32, text_dim=32,
                    out_dim=4, num_heads=4, num_layers=1).eval()


def _latent(c=4, f=3, h=8, w=8):
    torch.manual_seed(1)
    return [torch.randn(c, f, h, w)]


def test_forward_uniform_timestep(tiny_model):
    x = _latent()
    ctx = [torch.randn(5, 32)]
    with torch.no_grad():
        out = tiny_model(x, t=torch.tensor([500.0]), context=ctx, seq_len=48)
    assert out[0].shape == (4, 3, 8, 8)          # row 22: unpatchify
    assert out[0].dtype == torch.float32


def test_forward_per_token_timestep(tiny_model):
    """TI2V pattern: t = [B, seq_len] (row 7 of the shape map)."""
    x = _latent()
    ctx = [torch.randn(5, 32)]
    with torch.no_grad():
        out = tiny_model(x, t=torch.full((1, 48), 500.0), context=ctx,
                         seq_len=48)
    assert out[0].shape == (4, 3, 8, 8)


def test_forward_with_padding(tiny_model):
    """seq_len > true token count exercises the pad/crop invariant (row 12)."""
    x = _latent(f=3, h=8, w=8)  # L = 3*4*4 = 48
    ctx = [torch.randn(5, 32)]
    with torch.no_grad():
        out = tiny_model(x, t=torch.tensor([100.0]), context=ctx, seq_len=64)
    assert out[0].shape == (4, 3, 8, 8)


def test_forward_deterministic(tiny_model):
    x = _latent()
    ctx = [torch.randn(7, 32)]
    with torch.no_grad():
        a = tiny_model(x, t=torch.tensor([250.0]), context=ctx, seq_len=48)
        b = tiny_model(x, t=torch.tensor([250.0]), context=ctx, seq_len=48)
    assert torch.equal(a[0], b[0])


def test_i2v_variant_requires_y():
    model = WanModel(model_type="i2v", patch_size=(1, 2, 2), text_len=16,
                     in_dim=8, dim=64, ffn_dim=128, freq_dim=32, text_dim=32,
                     out_dim=4, num_heads=4, num_layers=1).eval()
    x = [torch.randn(4, 3, 8, 8)]
    y = [torch.randn(4, 3, 8, 8)]  # concatenated on channel dim inside
    ctx = [torch.randn(5, 32)]
    with torch.no_grad():
        out = model(x, t=torch.tensor([500.0]), context=ctx, seq_len=48, y=y)
    assert out[0].shape == (4, 3, 8, 8)
