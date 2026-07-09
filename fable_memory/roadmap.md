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

## Phase 2 — real-generation continuity hardening (needs CUDA machine)
Run WanTI2VEngine end-to-end · 2-shot continuity test with real frames ·
terminal-frame quality (anti-blur pick among last N frames) · overlap-frame
trimming at stitch · per-scene assemblies · quality thresholds tuned on real
output.

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
