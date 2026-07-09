# Roadmap

## Phase 1 — internal skeleton (THIS PR — complete)
Recon + tensor map · cognitive_fabric package (config/state/bus/registry/
runtime/hooks/logging) · 19 brains · memory layer (identity matrix, world,
temporal cache, sqlite, timeline, references) · reference token stack ·
conditioning/guidance modules · chunk scheduler · storyboard + longform
orchestrators with repair loop, disk streaming, checkpoints, resume ·
mock + real engine seams · tiled hierarchical scaler · stitcher · export
presets · CLI tasks · health check/installers · 160 CPU tests ·
5 upstream fixes (R-001..R-005).

## Phase 2A — continuity hardening, CPU-verifiable half (complete)
Terminal-frame quality selection (anti-blur pick + trim-to-anchor) ·
duplicated-lead-frame trimming at seams (recorded in timeline/ledger,
resume-safe) · user reference images into the engine path (--image opening
frame via i2v clamp, --ref_images into the token stack) · per-scene
assemblies · streaming finishing/upscale pass wired into export
(finish_video_file, explicit opt-in via --final_resolution).

## Phase 2B — real-generation continuity hardening (needs CUDA machine)
Run WanTI2VEngine end-to-end · 2-shot continuity test with real frames ·
quality thresholds tuned on real output · flash-attn parity re-confirmation.

## Phase 3 — identity/object/world memory on real output
Real reference encoder (open CLIP-style, local) for identity embeddings ·
drift detection on generated frames · approved-frame feedback into the
identity matrix · world drift via local VLM captioning.

## Phase 4 — attention-level conditioning (deepest engine work)
Context-token injection into cross-attention (hook + tests already in
fabric_hooks) · physics/collision attention bias · optional temporal K/V
snapshot in self-attention · per-expert conditioning for A14B MoE ·
fallback to original attention guaranteed by the R-002 dispatcher.

## Phase 5 — long-form at scale
Multi-hour resumable runs · per-act exports · editor brain (cut timing,
transition choice by emotion/motion) · beat-synced cuts from audio organ.

## Phase 6 — 4K finishing quality
Real-ESRGAN / VEnhancer backends behind HierarchicalScaler.register_backend ·
RIFE behind FrameInterpolator.register_backend · OCIO/ACES grading ·
anti-flicker temporal models.

## Phase 7 — anime & series
Style-sheet lock · anime validators · mouth-flap timing · opening/ending
templates · multi-episode continuity (SeriesBible persistence already works).

## Phase 8 — model factory & concurrency
LoRA key-mapping for WanModel · adapter training loops · reward model ·
multi-model VRAM scheduling (runner exists) · quantised variants.

## Installable app (parallel track)
windows_one_click_setup.ps1 (works) → packaged self-contained installer
(build_installer.ps1 placeholder) → desktop shell only after the engine
phases stabilise, and strictly as a thin skin over generate.py.
