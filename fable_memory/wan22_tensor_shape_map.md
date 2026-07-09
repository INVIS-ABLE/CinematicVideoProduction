# Wan 2.2 Tensor Shape Map

Symbolic dims: `B` batch (pipelines use B=1 lists), `F` output frames
(4n+1), `F_lat = (F-1)/4 + 1` latent frames, `H×W` pixels,
`(sT,sH,sW)` = vae_stride — **(4,8,8) A14B / (4,16,16) TI2V-5B**,
`H_lat = H/sH`, `W_lat = W/sW`, patch `(pT,pH,pW) = (1,2,2)`,
`L = F_lat · H_lat/pH · W_lat/pW` (token count, padded up to `seq_len`),
`D` = model dim (5120 A14B / 3072 5B), `N` heads (40 / 24), `d = D/N` head dim
(128), `z` = VAE latent channels (**16** A14B / **48** 5B).

Worked example — TI2V-5B, 1280×704, 121 frames:
`F_lat = 31`, `H_lat = 44`, `W_lat = 80`, `L = 31·22·40 = 27,280`.

| # | Stage | File:anchor | Shape | dtype |
| --- | --- | --- | --- | --- |
| 1 | Text embedding out (T5) | `t5.py:506` | list of `[L_txt_i ≤ 512, 4096]` | bf16 |
| 2 | Context padded in DiT | `model.py:473` | `[B, 512, 4096]` → proj `[B, 512, D]` | bf16/fp32 |
| 3 | Image ref pixel in (TI2V i2v) | `textimage2video.py:477` | `[3, 1, H, W]` (unsqueezed frame axis) | fp32 |
| 4 | Image ref latent (VAE enc) | `textimage2video.py:512` | `[z, 1, H_lat, W_lat]` | fp32 |
| 5 | Noise latent init | `text2video.py:279` / `textimage2video.py:311` | `[z, F_lat, H_lat, W_lat]` | fp32 |
| 6 | Latent clamp masks | `utils.py:172 masks_like` | same as #5; `mask2[:,0]=0` locks frame 0 | fp32 |
| 7 | Timestep tensor | `text2video.py:337` | `[B]`; TI2V per-token: `[B, seq_len]` (`textimage2video.py:573-578`) | long/fp32 |
| 8 | Sinusoidal t-embed | `model.py:14` | `[B·seq_len, 256]` → unflatten `[B, seq_len, 256]` | fp32 (from fp64) |
| 9 | Time embedding e | `model.py:465` | `[B, seq_len, D]` | fp32 (asserted) |
| 10 | Time projection e0 | `model.py:468` | `[B, seq_len, 6, D]` | fp32 (asserted) |
| 11 | Patch embedding out | `model.py:448` | `[1, D, F_lat, H_lat/2, W_lat/2]` per sample | model dtype |
| 12 | Token sequence x | `model.py:451-457` | `[B, seq_len, D]` (zero-padded from L) | model dtype |
| 13 | grid_sizes | `model.py:449` | `[B, 3]` = (F_lat, H_lat/2, W_lat/2) | long |
| 14 | RoPE freqs buffer | `model.py:400` | `[1024, d/2]` complex64 (fp64 math), split (d/2−2⌊d/6⌋, ⌊d/6⌋, ⌊d/6⌋) over (F,H,W) | complex |
| 15 | Self-attn q/k/v | `model.py:138-141` | `[B, seq_len, N, d]` | model dtype |
| 16 | rope_apply out | `model.py:39-66` | `[B, seq_len, N, d]` | fp32 |
| 17 | flash/SDPA attn out | `attention.py:24-179` | `[B, seq_len, N, d]` → flatten(2) `[B, seq_len, D]` | in dtype |
| 18 | Cross-attn q | `model.py:170` | `[B, seq_len, N, d]` | model dtype |
| 19 | Cross-attn k/v (context) | `model.py:171-172` | `[B, 512, N, d]` | model dtype |
| 20 | Block out / residual | `model.py:219-259` | `[B, seq_len, D]` (adaLN e chunks: 6 × `[B, seq_len, 1, D]`) | mixed, residual fp32 adds |
| 21 | Head out | `model.py:279` | `[B, seq_len, pT·pH·pW·z_out]` (z_out = out_dim) | fp32 |
| 22 | unpatchify out | `model.py:499-522` | list of `[z_out, F_lat, H_lat, W_lat]` | fp32 |
| 23 | MoE expert selection | `text2video.py:186` | scalar boundary test `t ≥ boundary·1000` → picks whole DiT (high noise ≥, low noise <) — same shapes both experts | — |
| 24 | CFG combine | `text2video.py:351` | `[z_out, F_lat, H_lat, W_lat]` | fp32 |
| 25 | Scheduler step in/out | `text2video.py:354-360` | `[1, z, F_lat, H_lat, W_lat]` ↔ squeeze | fp32 |
| 26 | VAE decode in | `vae2_2.py:1038` | list of `[z, F_lat, H_lat, W_lat]` | fp32 |
| 27 | VAE decode out (video) | pipelines' return | `[3, F, H, W]` in [-1, 1] | fp32 |
| 28 | save_video input | `generate.py:551` | `[1, 3, F, H, W]` (`video[None]`) | fp32 |

## Invariants every fabric modification must preserve

1. `frame_num ≡ 1 (mod 4)` — VAE temporal stride.
2. `H ≡ 0 (mod sH·pH)`, `W ≡ 0 (mod sW·pW)` (16 for A14B, 32 for TI2V-5B).
3. `seq_lens.max() <= seq_len` assertion (`model.py:453`); anything appended
   to the *video token* sequence changes `seq_len` and RoPE grid mapping —
   video-token injection is therefore **not** the Phase-4 route; context
   (cross-attn) injection is, because `context` length is free (`[B, L2, C]`,
   k_lens optional).
4. adaLN `e`/`e0` must remain fp32 (`assert e.dtype == torch.float32`).
5. RoPE is computed from `grid_sizes`, not from `seq_len` — padded tail tokens
   receive no rotation and are cropped by `unpatchify` (`u[:math.prod(v)]`).
6. Timestep may be scalar-per-sample or per-token; fabric continuity code must
   emit per-token `t` when using latent clamping (TI2V pattern).
7. Attention I/O contract: in `[B, L, N, d]` → out `[B, L, N, d]`; dispatcher
   fallback transposes internally to `[B, N, L, d]` for SDPA and transposes
   back. Device may be CPU (SDPA) or CUDA (flash); dtype in == dtype out.

## Verified-by-test shapes

`tests/cognitive_fabric/test_tensor_shapes.py` runs a real tiny
`WanModel(dim=64, num_heads=4, num_layers=1, in_dim=4, out_dim=4)` forward on
CPU through the SDPA fallback and asserts rows 7, 12, 21, 22 (both `t=[B]` and
per-token `t=[B, seq_len]`).
`test_wan_forward_smoke.py` asserts the attention dispatcher I/O contract
(row 17) and dtype/device preservation.
