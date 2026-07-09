# Wan 2.2 Cognitive Fabric — Engine Recon

Produced from direct inspection of the vendored Wan 2.2 source at upstream commit
`42bf4cfaa384bc21833865abc2f9e6c0e67233dc` (github.com/Wan-Video/Wan2.2, Apache-2.0).
Nothing in this document is guessed; every claim has a file/line anchor.

## 1. Repository tree (source files)

```
generate.py                      # single CLI entry point for all tasks (575 lines)
wan/__init__.py                  # exports WanT2V, WanI2V, WanTI2V, WanS2V, WanAnimate
wan/configs/                     # EasyDict model configs per task
  shared_config.py               # t5 settings, param_dtype=bf16, fps=16, neg prompt, frame_num=81
  wan_t2v_A14B.py                # MoE: low/high noise ckpts, boundary=0.875, vae_stride (4,8,8)
  wan_i2v_A14B.py                # MoE: boundary=0.900
  wan_ti2v_5B.py                 # single model, vae_stride (4,16,16), fps=24, frame_num=121
  wan_s2v_14B.py                 # speech-to-video
  wan_animate_14B.py             # character animation/replacement
wan/text2video.py                # WanT2V pipeline (MoE dual model)
wan/image2video.py               # WanI2V pipeline (MoE dual model, first-frame conditioning)
wan/textimage2video.py           # WanTI2V pipeline (5B single model, t2v() + i2v() paths)
wan/speech2video.py              # WanS2V pipeline (audio-driven, chunked autoregressive)
wan/animate.py                   # WanAnimate pipeline (pose/face driven, LoRA relighting)
wan/modules/model.py             # WanModel — the DiT backbone (all tasks)
wan/modules/attention.py         # flash_attention (FA2/FA3) + attention() SDPA dispatcher
wan/modules/t5.py                # umT5-xxl text encoder (T5EncoderModel)
wan/modules/tokenizers.py        # HuggingFace tokenizer wrapper
wan/modules/vae2_1.py            # Wan2.1 VAE (z_dim=16, stride 4,8,8) — used by A14B + s2v/animate
wan/modules/vae2_2.py            # Wan2.2 VAE (z_dim=48, stride 4,16,16) — used by TI2V-5B
wan/modules/s2v/                 # audio encoder (wav2vec2), S2V model variant
wan/modules/animate/             # animate model, motion/face encoders, pose preprocess
wan/distributed/                 # FSDP sharding, ulysses sequence parallel
wan/utils/fm_solvers.py          # FlowDPMSolverMultistepScheduler (flow matching DPM++)
wan/utils/fm_solvers_unipc.py    # FlowUniPCMultistepScheduler (default)
wan/utils/prompt_extend.py       # QwenPromptExpander (local), DashScopePromptExpander (cloud)
wan/utils/utils.py               # save_video, masks_like, best_output_size, merge_video_audio
```

## 2. Model entry points and generation commands

One CLI: `generate.py`. Tasks map 1:1 to `WAN_CONFIGS` keys
(`wan/configs/__init__.py:13-19`): `t2v-A14B`, `i2v-A14B`, `ti2v-5B`,
`animate-14B`, `s2v-14B`.

Exact function call chain, prompt → video (T2V path):

```
generate.py:_parse_args → _validate_args        (defaults from cfg; seed randomised if <0)
generate.py:generate(args)
  wan.WanT2V.__init__                            (text2video.py:33)
    T5EncoderModel(...)                          (t5.py:472 — encoder-only umT5-xxl)
    Wan2_1_VAE(...)                              (vae2_1.py)
    WanModel.from_pretrained(subfolder=low_noise_checkpoint)   # diffusers ModelMixin loader
    WanModel.from_pretrained(subfolder=high_noise_checkpoint)
  wan.WanT2V.generate(prompt, size, frame_num, shift, solver, steps, guide_scale, seed, offload)
    target_shape = (16, (F-1)/4+1, H/8, W/8)     (text2video.py:253)
    seq_len = H/8 * W/8 / 4 * T_lat              (text2video.py:257)
    context = text_encoder([prompt]); context_null = text_encoder([n_prompt])
    noise = randn(target_shape)                  (seeded torch.Generator on device)
    scheduler = FlowUniPC | FlowDPM              (set_timesteps(steps, shift))
    per timestep t:
      model = high_noise if t >= boundary*1000 else low_noise   (text2video.py:169-201)
      noise_pred = uncond + cfg * (cond - uncond)               (dual forward, CFG)
      latents = scheduler.step(noise_pred, t, latents)
    videos = vae.decode(latents)                 (rank 0 only)
generate.py: save_video(tensor, fps=cfg.sample_fps)             (utils.py:90)
```

TI2V-5B (`textimage2video.py`) differs: single `self.model`; `generate()`
routes to `.t2v()` (no image) or `.i2v(img)` (image); i2v encodes the image
with the 2.2 VAE and **clamps the first latent frame every step**:

```
mask1, mask2 = masks_like([noise], zero=True)    # mask2[:,0] = 0 → frame-0 locked
latent = (1-mask2) * z_img + mask2 * latent      (textimage2video.py:551, re-applied :598)
timestep is per-token: t=0 for locked frame-0 tokens, t elsewhere (:573-578)
```

**This latent-clamp mask + per-token timestep mechanism is the native seam for
first-frame lock, last-frame handoff, and chunk-to-chunk continuity.** The
fabric's continuity conditioning builds on it rather than inventing a new one.

S2V (`speech2video.py`) already generates **chunked autoregressive video**
(`num_repeat` clips, motion carried via `motioner`) — internal precedent for
long-form chunking. Animate (`animate.py`) has pose retarget preprocess,
CLIP-based face conditioning, and an optional relighting LoRA loaded via peft.

## 3. Model loading / checkpoint logic

- `WanModel.from_pretrained(checkpoint_dir[, subfolder])` — diffusers
  `ModelMixin`; safetensors under the checkpoint dir.
- A14B tasks hold **two** full DiTs (`low_noise_model`, `high_noise_model`) and
  swap them between GPU/CPU per timestep boundary
  (`text2video.py:_prepare_model_for_timestep`). This is the Wan 2.2 "MoE":
  expert-per-noise-band, not per-token routing.
- T5 loads from `models_t5_umt5-xxl-enc-bf16.pth`, tokenizer `google/umt5-xxl`.
- VAE loads from a single `.pth` (`Wan2.1_VAE.pth` / `Wan2.2_VAE.pth`).
- No unified model registry, no quantised variants, no LoRA loader in the base
  repo (peft only inside animate's relighting LoRA).

## 4. Prompt encoding

- `T5EncoderModel.__call__(texts, device)` (`t5.py:506`) → list of `[L_i, 4096]`
  tensors trimmed to true lengths.
- `WanModel.forward` re-pads each context to `text_len=512` and projects
  4096 → dim via `text_embedding` MLP (`model.py:473-478`).
- Negative prompt default is a Chinese boilerplate string in
  `shared_config.py` (`sample_neg_prompt`).
- `prompt_extend.py` offers a local Qwen or a DashScope-cloud expander —
  **cloud path violates local-first; fabric never uses DashScope.**

## 5. DiT internals (`wan/modules/model.py`)

- `patch_embedding = Conv3d(in_dim, dim, k=stride=patch_size)`; patch (1,2,2).
- `sinusoidal_embedding_1d(freq_dim=256, t)` → `time_embedding` MLP → e;
  `time_projection` → e0 `[B, L, 6, dim]` (6-way adaLN modulation per block).
- `t` may be `[B]` (uniform) or `[B, seq_len]` (per-token — used by TI2V i2v).
- 3D RoPE: `freqs` split across (frame, height, width) sub-dims
  (`rope_params`/`rope_apply`, float64 complex math, `model.py:27-66`).
- `WanAttentionBlock` = adaLN-modulated self-attn (RoPE) → cross-attn (text)
  → FFN. `Head` projects to `out_dim * prod(patch_size)`; `unpatchify` restores
  `[C_out, F_lat, H_lat, W_lat]`.
- Weights: Xavier init; head zero-init.
- **Cross-attention takes any `[B, L2, C]` context** — reference/identity/world
  tokens can be appended to text context without touching layer weights. This
  is the fabric's conditioning insertion point (Phase 4).

## 6. Attention (`wan/modules/attention.py`)

- `flash_attention()` requires CUDA + flash-attn 2/3 (`assert q.device.type ==
  'cuda'`, `assert FLASH_ATTN_2_AVAILABLE`).
- `attention()` dispatcher falls back to
  `torch.nn.functional.scaled_dot_product_attention` when flash-attn is absent
  (padding masks dropped with a warning).
- **Found defect/limitation:** `model.py` imported `flash_attention` directly,
  bypassing the dispatcher — so the base DiT could not run without flash-attn
  even though a fallback exists in the same package. Fixed in this refactor
  (see `fable_memory/refactor_log.md` R-002): `model.py` now routes through
  `attention()`. Behaviour with flash-attn installed is unchanged (dispatcher
  forwards identical args); without it, SDPA fallback enables CPU/any-GPU
  smoke tests of the real forward pass. `s2v`/`animate` model variants still
  import `flash_attention` directly — left untouched in Phase 1.

## 7. Scheduler / sampling

- Flow-matching UniPC (`fm_solvers_unipc.py`, default) and DPM++
  (`fm_solvers.py`); `shift` warps the sigma schedule (per-task defaults:
  T2V 12.0, I2V/TI2V 5.0).
- CFG doubles the forward count; A14B uses `(low, high)` guide-scale tuple.
- Sampling loop lives in each pipeline, not in the model — the fabric's
  generation controller wraps pipelines, not the scheduler.

## 8. Distributed / memory / precision

- FSDP sharding (`distributed/fsdp.py`) for T5 and DiT; ulysses sequence
  parallel monkey-patches `self_attn.forward` and `model.forward`
  (`text2video.py:150-154`) — **any attention-signature change must keep
  `sp_attn_forward` compatible** (it is untouched in Phase 1).
- Offload strategy: `offload_model=True` (single GPU default) moves inactive
  MoE expert / text encoder to CPU each step; `t5_cpu`, `init_on_cpu` flags.
- Precision: autocast bf16 around sampling; adaLN modulation asserted fp32;
  RoPE in fp64. `convert_model_dtype` converts DiT params to bf16.

## 9. Frame decode & output

- `vae.decode(latents)` → `[3, F, H, W]` in [-1, 1];
  `save_video` (`utils.py:90`) → mp4 at `cfg.sample_fps` (16 A14B / 24 TI2V).
- S2V muxes audio via `merge_video_audio` (ffmpeg).

## 10. CLI argument surface (`generate.py`)

`--task --size --frame_num --ckpt_dir --offload_model --ulysses_size --t5_fsdp
--t5_cpu --dit_fsdp --save_file --prompt --use_prompt_extend
--prompt_extend_* --base_seed --image --sample_solver --sample_steps
--sample_shift --sample_guide_scale --convert_model_dtype` + animate/s2v extras.
Frame count constraint: **`frame_num` must be 4n+1** (VAE temporal stride 4).

## 11. Existing limitations (verified, not assumed)

1. `import wan` unconditionally imports `speech2video` + `animate`, pulling
   librosa/decord/cv2/peft even for T2V-only use; base `requirements.txt`
   does not include them (they live in `requirements_s2v.txt` /
   `requirements_animate.txt`) → base install crashed on import.
   **Fixed via lazy `__getattr__` exports (refactor R-001).**
2. No memory of any kind between `generate()` calls — no character, object,
   world, or temporal state. Every clip is amnesiac.
3. No storyboard/multi-shot concept; one prompt → one clip.
4. Max practical clip ≈ 5 s (81 frames @16fps A14B, 121 @24fps TI2V).
5. No continuation API: TI2V's first-frame clamp exists but no
   last-frame→next-chunk plumbing.
6. No quality validation, no repair loop, no upscaling, no stitching.
7. No seed strategy beyond a single int; no per-shot control.
8. Hard flash-attn dependency in the DiT (fixed, see §6).
9. Prompt extension optionally calls a paid cloud API (DashScope).
10. No model registry/adapter (LoRA) system for the main tasks.

## 12. Files safe to refactor first vs. files not to break

Safe first (Phase 1 touched):
- `wan/__init__.py` (lazy exports — API-preserving)
- `wan/modules/model.py` (attention dispatch only — numerically identical
  under flash-attn)
- `generate.py` (additive flags/tasks only)
- new package `wan/cognitive_fabric/**` (pure addition)

Must not break (untouched in Phase 1):
- `wan/modules/attention.py` internals (sp_attn_forward depends on
  `flash_attention` semantics)
- `wan/distributed/**` (FSDP/ulysses)
- `wan/utils/fm_solvers*.py` (scheduler math)
- VAE implementations, T5, tokenizers
- s2v/animate model files and preprocessing

## 13. GPU memory hotspots

1. Dual A14B experts resident (mitigated by per-timestep offload swap).
2. CFG dual forward per step (2× activation traffic).
3. `seq_len` grows linearly with F_lat × H_lat × W_lat — 720p×121f TI2V is
   ~34,650 tokens (see tensor map §latents); attention is O(L²) without flash.
4. VAE decode of full clip at once (`vae.decode(x0)`).
5. T5 umT5-xxl (~11 GB bf16) unless `t5_cpu`.

## 14. Cognitive fabric hook points (chosen)

| Hook | Where | Phase |
| --- | --- | --- |
| Task router | `generate.py` cognitive-* tasks → `fabric_runtime` | 1 (done) |
| Storyboard → chunks | new `pipeline/storyboard_orchestrator.py` + `chunk_scheduler.py` | 1 (done) |
| Prompt compile | new `pipeline/prompt_compiler.py` feeding pipeline `generate(input_prompt=…, n_prompt=…)` | 1 (done) |
| First-frame continuity | TI2V `.i2v(img=terminal_frame)` latent clamp | 2 |
| Terminal-frame capture | post-`vae.decode` in generation controller | 2 |
| Reference/identity tokens | append to cross-attn `context` (dim 4096 pre-projection) | 4 |
| Physics/collision bias | additive attention bias in `WanSelfAttention` | 4 |
| Temporal KV | optional K/V snapshot in self-attn | 4 |
| MoE routing metadata | guidance-strength + prompt emphasis per noise band (A14B boundary switch) | 4 |
