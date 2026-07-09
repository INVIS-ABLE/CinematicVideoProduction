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
