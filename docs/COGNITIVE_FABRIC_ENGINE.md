# Wan 2.2 Cognitive Fabric Engine

The Cognitive Fabric is an **internal upgrade of Wan 2.2**, not a wrapper:
the base model remains the generative heart; the fabric gives it the missing
organs — storyboard understanding, character/object/world memory, long-form
chunking with continuity, quality validation, a repair loop, progressive
finishing, and a strict local-first policy.

Activation is opt-in. With `--cognitive_fabric false` (default) and a classic
task, upstream Wan behaviour is byte-for-byte preserved.

## Quick start

```bash
# environment check (no GPU or weights needed)
python -m wan.cognitive_fabric.install.health_check

# plan a short film from one prompt — no frames generated
python generate.py --task cognitive-short \
  --prompt "A damaged hero walks through neon rain in a futuristic city at night" \
  --duration_seconds 30 --dry_run true

# run the full pipeline with the mock engine (any machine)
python generate.py --task cognitive-short --prompt "..." --duration_seconds 20

# real generation (CUDA GPU + downloaded checkpoint)
python generate.py --task cognitive-short --prompt "..." \
  --duration_seconds 30 --ckpt_dir ./models/Wan2.2-TI2V-5B

# full storyboard film
python generate.py --task cognitive-film \
  --storyboard ./projects/my_film/storyboard.json \
  --ckpt_dir ./models/Wan2.2-TI2V-5B --duration_minutes 60 \
  --base_resolution 720p --final_resolution 4k
```

Original Wan tasks are untouched:

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./models/Wan2.2-TI2V-5B --prompt "..."
```

## Architecture

```
generate.py ── cognitive-* tasks ──▶ FabricRuntime
                                        │
     ┌────────────── brains ────────────┼──────── memory ─────────────┐
     │ director · cinematographer ·     │  GlobalIdentityMatrix       │
     │ lighting · motion · physics ·    │  WorldMemory · ObjectMemory │
     │ prompt · continuity · quality ·  │  TemporalKVCache            │
     │ repair · resource · sound · …    │  TimelineMemory · MemoryDB  │
     └───────────────┬──────────────────┴──────────────┬──────────────┘
                     ▼                                 ▼
        StoryboardOrchestrator ◀── conditioning (tokens/guidance/clauses)
                     │  per chunk: compile → generate → validate → repair
                     ▼
        VideoEngine seam ── WanTI2VEngine (real) / MockWanEngine (dry)
                     │            └── wan.WanTI2V.generate() — the actual
                     ▼                Wan 2.2 sampling loop & VAE
        chunks/ → stitcher → HierarchicalScaler → FinishingPipeline → exports/
```

Brains communicate through shared project memory and the message bus; every
brain is registered in `FabricRegistry` and individually replaceable.

## Continuity model (what makes clips connect)

1. **First-frame lock** — Wan TI2V's own latent clamp (`masks_like` +
   per-token timesteps) conditions each continuation chunk on the previous
   chunk's terminal frame.
2. **Identity matrix** — characters/objects registered once, re-conditioned
   into every shot's prompt (and, in Phase 4, into cross-attention context).
3. **World memory** — location/time/weather/light-direction locked per scene;
   drift is detected field-by-field.
4. **Temporal cache** — terminal frames + compressed embeddings + token
   snapshots carried chunk-to-chunk with decay; seam similarity is measured
   and fed to the quality report.
5. **Continuity brain** — classifies transitions (continuation, hard cut,
   location change) and emits continuity clauses per shot.

## Honest capability statement

| Capability | Status |
| --- | --- |
| Storyboard → chunk plan → generation → stitch → export | working (real engine on CUDA+ckpt, mock elsewhere) |
| 60-minute film **planning**, disk streaming, crash-resume | working, tested |
| 60-minute film in one model call | never claimed — chunked by design |
| Flicker/motion/sharpness/exposure/seam scoring | working (pixel metrics) |
| Face/hand/anatomy/prompt-match scoring | interface only — declines to score until local validator models land |
| Physics | expectation + prompt guidance + repair hooks; **not** simulation |
| 4K | progressive tiled upscale (bicubic today; ESRGAN/VEnhancer hooks) |
| MoE routing | guide-scale shaping + metadata; deeper routing is Phase 4 |
| LoRA/adapter training | registry + policy today; training loops Phase 8 |

See `fable_memory/roadmap.md` for the phase plan and
`fable_memory/refactor_log.md` for every internal engine change.
