# Regression Log

Gate: MAKE WAN RUN AFTER EVERY CHANGE.

## 2026-07-09 — Phase 1 complete
- `python -m pytest tests/cognitive_fabric/` → **160 passed** (CPU, no
  weights, no CUDA, no flash-attn).
- Real `WanModel` tiny forward on CPU via SDPA fallback: PASS (uniform and
  per-token timesteps, padding, i2v variant, determinism).
- `import wan` on base deps: PASS (R-001/R-003/R-005).
- Original CLI guard: `--task t2v-A14B` without ckpt_dir asserts exactly as
  upstream ("Please specify the checkpoint directory."): PASS.
- End-to-end mock run via CLI (`cognitive-short`, 12 s): 4 chunks generated,
  stitched draft_assembly.mp4, checkpoints written: PASS.
- Crash-resume: immediate rerun regenerates 0 chunks, skips 4: PASS.
- 60-minute storyboard dry-run: 700+ chunks planned, within budget: PASS.

Not verified in this environment (no CUDA GPU / no model weights here):
- Real Wan sampling through WanTI2VEngine (code path exercised only up to
  the CUDA guard). Guarded by explicit runtime errors + engine fallback.
- flash-attn numerical parity (dispatcher forwards identical args; needs a
  CUDA machine to re-confirm end-to-end).

## 2026-07-09 — Phase 2A complete (CPU-verifiable half of Phase 2)
- `python -m pytest tests/cognitive_fabric/` → **167 passed** (CPU, no
  weights, no CUDA, no flash-attn).
- Terminal-frame quality selection: sharpest-of-window picker verified
  against deliberately blurred tails; clips trimmed to end on the chosen
  anchor frame.
- Seam hygiene: continuation chunks drop the duplicated conditioning frame;
  lead/tail trims recorded in timeline + ledger and preserved across resume.
- User reference path: `--image` conditions the opening chunk (i2v clamp),
  `--ref_images` feeds the token stack; missing files fail loudly.
- Per-scene assemblies written to exports/scenes/<scene_id>.mp4.
- Streaming finishing pass (`finish_video_file`): batch-streamed read →
  tiled upscale → look → H.264 write; frame count preserved; only runs when
  `--final_resolution` is explicitly given.
- CLI end-to-end: cognitive-short with --image/--ref_images/--final_resolution
  produced draft_assembly.mp4, scenes/scene_001.mp4, final_480p.mp4: PASS.

Still requires a CUDA machine (unchanged): real WanTI2VEngine sampling,
flash-attn parity, threshold tuning on real output.
