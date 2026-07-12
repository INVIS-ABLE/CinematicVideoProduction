# Cinematic Studio Local — Engineering Blueprint

**Status:** architecture specification and delivery contract  
**Target:** local workstation 1.0  
**Branch:** `agent/local-cinematic-studio`  
**Hard reference limits:** 15 images, 3 videos, 3 voice/audio samples

> This specification defines the system that must be built. It does not claim that every subsystem is already implemented. The current branch is a useful UI and packaging scaffold, but it is not production-ready until the acceptance gates in Chapter 5 pass.

## Product definition

Cinematic Studio Local is a modular, private application for creating short clips and persistent multi-shot stories from text, images, video, voice, and editable 3D worlds. It combines:

1. AI video generation and visual refinement.
2. Authoritative 3D scene state, cameras, lighting, collision, and physics.
3. Persistent story memory for characters, worlds, props, voices, scenes, shots, and continuity.
4. Professional audio, stitching, quality control, and verified 4K delivery.

AI-generated pixels are not the source of truth for geometry or physics. The canonical world is a versioned OpenUSD stage plus structured project state. A physics engine resolves and bakes motion. AI models may infer, propose, reconstruct, generate, animate, or refine assets, but they cannot silently overwrite authoritative state.

## Non-negotiable requirements

| Area | Contract |
| --- | --- |
| Local-first | Runs offline after installation and selected model-pack downloads. No outbound connection without explicit opt-in. |
| Hardware-scaled | Fast probe exposes only eligible models, render modes, and quality settings. |
| Crash containment | UI and project database survive CUDA, Blender, connector, codec, and worker failures. |
| Modularity | Heavy capabilities are isolated, versioned workers with health checks and manifests. |
| Persistent world | Geometry, transforms, materials, cameras, lights, physics state, and damage persist. |
| Persistent story | Character identity, wardrobe, voice, dialogue, relationships, and continuity persist. |
| Physics | Collision and dynamics are engine-computed, inspectable, validateable, and bakeable. |
| 4K | Native 4K from the 3D path; verified 4K finishing from neural paths. |
| Audio | Dialogue, consented character voice, lip sync, ambience, foley, music, stems, and subtitles. |
| Resumability | Every expensive operation is a checkpointed DAG node with cancellation and retry. |
| Provenance | Outputs record source hashes, model versions, settings, seeds, licences, and consent. |
| Packaging | One-click desktop installer; model packs are independent, versioned downloads. |

## Reference architecture

```text
Desktop supervisor (Tauri/Rust target; Python launcher during development)
  └─ loopback IPC and per-launch token
      └─ Control plane
          ├─ project/story APIs
          ├─ hardware and capability planner
          ├─ resumable DAG scheduler
          ├─ model/connector registry
          └─ provenance and quality coordinator
              ├─ AI workers: Wan, LTX, VLM, segmentation, reconstruction
              ├─ World worker: OpenUSD, Blender, PhysX/Bullet/Warp
              ├─ Audio worker: TTS, voice, alignment, lip sync, mixer
              └─ Finish worker: FFmpeg, VSR, interpolation, colour, QC

Data plane
  ├─ SQLite WAL metadata and event log
  ├─ OpenUSD stages and simulation caches
  ├─ content-addressed media/mesh/audio artifact store
  └─ model packs, render cache, logs, checksums, and backups
```

The API/control plane must never load a heavy diffusion model. The desktop shell, API, GPU model workers, 3D/physics worker, audio worker, and finishing worker are separate operating-system processes. The default is one diffusion job per GPU; additional concurrency is enabled only by the hardware planner.

## Specification chapters

1. [Hardware, runtime, model packs, and scheduling](blueprint/01-hardware-runtime.md)
2. [World understanding, OpenUSD, 3D assets, physics, lighting, and rendering](blueprint/02-world-physics-rendering.md)
3. [Persistent storytelling, character voice, audio, stitching, and 4K](blueprint/03-story-audio-finishing.md)
4. [Connectors, local API, packaging, security, and observability](blueprint/04-connectors-packaging-security.md)
5. [Testing, acceptance criteria, current gaps, and delivery roadmap](blueprint/05-testing-roadmap.md)

## Machine-readable contracts

- [Hardware profile schema](schemas/hardware_profile.schema.json)
- [Model-pack manifest schema](schemas/model_manifest.schema.json)
- [World-state schema](schemas/world_state.schema.json)
- [Shot/take specification schema](schemas/shot_spec.schema.json)

## Definition of complete

The application is complete only when it can inspect a supported workstation, choose a safe pipeline, build or load a persistent world, simulate and validate physical motion, render or generate a shot, give characters consented voices, preserve continuity across shots, recover from worker failure, stitch and mix the story, verify a real 4K master, reopen the production later, and explain every source, model, setting, and decision used.