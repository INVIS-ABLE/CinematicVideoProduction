# Long-Form (60-Minute) Pipeline

Honest statement first: **no model generates 60 minutes in one call.** The
fabric produces long-form output as a structured hierarchy —

```
Film → Acts → Scenes → Shots → Chunks (≈5 s, 4n+1 frames) → Frames
```

## How it works today (tested)

1. `LongformOrchestrator.film_plan()` expands a storyboard into the full
   chunk plan (a 60-minute storyboard → 700+ chunks) and checks the duration
   budget. Covered by `tests/cognitive_fabric/test_longform_chunking.py`.
2. Chunks generate sequentially; each opens on its predecessor's terminal
   frame and inherits identity/world/temporal memory.
3. Every approved chunk streams to `projects/<id>/chunks/` — frames are never
   accumulated in RAM; the working set is one chunk.
4. `FabricState` checkpoints after every chunk. A crash at minute 43 resumes
   from the last approved chunk (`resume_or_create` + ledger skip) — tested.
5. Stitching is progressive: chunks → draft assembly now; per-scene and
   per-act assemblies as the editor brain lands.
6. Finishing runs per clip through the tiled hierarchical scaler
   (720p → 1080p → 1440p → 4K) and export presets.

## Project folder layout

```
projects/<project_id>/
  storyboard.json  film_plan.json  timeline.json  run_report.json  memory.db
  chunks/            # per-chunk mp4 + terminal-frame png
  scenes/ shots/ frames_720p/ frames_1080p/ frames_1440p/ frames_4k/
  audio/ subtitles/ exports/ logs/ checkpoints/
```

## Current limits (kept honest)

- Real-engine long-form throughput is bounded by Wan sampling speed
  (~minutes per 5 s chunk on a single consumer GPU).
- Seam quality across chunks uses first-frame conditioning + seam-similarity
  scoring; deeper temporal K/V injection is Phase 4.
- Scene-level parallel generation across multiple GPUs is Phase 8
  (concurrent model runner already schedules by VRAM budget).
