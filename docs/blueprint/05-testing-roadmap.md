# Chapter 5 — Testing, Acceptance Criteria, Current Gaps, and Delivery Roadmap

## 1. Test strategy

The application needs unit, contract, integration, golden-output, performance, chaos, installer, migration, and end-to-end tests. A green mock-only test suite is useful but is not evidence that CUDA/model-weight production paths work.

### Schema and storage tests

- every JSON schema accepts current fixtures and rejects malformed/unknown fields where required;
- SQLite migrations are forward/backward tested with backup/rollback;
- WAL concurrency, transaction rollback, crash recovery, and integrity checks;
- path traversal, symlink, archive bomb, and upload-size tests;
- artifact SHA-256 deduplication, lease, garbage collection, and approved-artifact retention;
- OpenUSD layer/version/reference round trips;
- coordinate/unit conversion fixtures.

### Hardware and model tests

- fixtures for CPU-only, NVIDIA, AMD, Intel, Apple, single/multi-GPU, low disk, unsupported drivers;
- probe timeouts and partial-information handling;
- model eligibility and exact blocked reasons;
- manifest licence/hash/signature failures;
- worker launch, health, smoke inference, cancellation, timeout, and cleanup;
- OOM ladder and resource-lock behavior;
- one-diffusion-job-per-GPU enforcement.

### 3D and physics tests

- known geometry import/scale/material/collision fixtures;
- deterministic falling body, stack, hinge, rolling/sliding, high-speed CCD, and camera-collision scenes;
- penetration/contact tolerances and cache hash invalidation;
- physics bake reopening on another session;
- camera/lens/exposure/light metadata round trip;
- control-pass frame count and object-ID consistency.

### Story and audio tests

- character/world/prop/voice persistence after close/reopen;
- version boundaries and dependency invalidation;
- terminal-frame and terminal-physics continuity packet;
- consent required for voice references;
- pronunciation/alignment/viseme timing;
- 48 kHz stem generation, loudness, true peak, silence, clipping, and A/V sync;
- subtitle timing;
- two-shot and multi-scene story assembly.

### Video and finishing tests

- clip timebase/frame-rate/audio-layout normalization;
- cuts, dissolves, crossfades, missing-audio padding, and seam QC;
- VSR/interpolation failure fallback;
- exact 1920×1080, 3840×2160, and 4096×2160 outputs;
- codec/profile/bit-depth/pixel-format/colour metadata;
- decode sweep, black/frozen/duplicate frames, tail truncation, and duration;
- master checksum/provenance.

### Packaging tests

- fresh supported Windows VM install, first launch, safe mode, repair, uninstall;
- no preinstalled Python required;
- app starts offline;
- FFmpeg notices and model licences present;
- model-pack download interruption/resume/hash failure;
- application update, rollback, and project migration;
- non-admin install where supported;
- antivirus/SmartScreen signing path and crash cleanup.

## 2. Chaos and failure tests

During an active job, automated tests terminate or break:

- CUDA/model worker;
- Blender/physics worker;
- FFmpeg/finish worker;
- connector process or LAN worker;
- disk writes and temp volume;
- network during a model-pack download;
- application/control-plane process;
- power/restart simulation at node boundaries.

Expected behavior:

- UI/supervisor remains alive when a child worker dies;
- project database stays consistent;
- GPU/file/worker locks are released;
- partial outputs remain invalid and isolated;
- successful upstream artifacts remain valid;
- restart offers Resume from the last valid DAG node;
- failure message identifies component, cause, and safe action.

## 3. Release acceptance criteria

A 1.0 release must prove all of the following:

1. Fresh installer launches offline on a clean supported machine.
2. Hardware scan completes without loading a generative model.
3. Every blocked model has a precise, actionable reason.
4. Wan TI2V is not offered when its tested requirement is not met.
5. Killing a GPU worker does not terminate the UI or corrupt the project.
6. A job resumes after application restart from the last valid DAG node.
7. Reopening a project restores characters, voices, worlds, shots, simulation caches, and approvals.
8. A two-shot story preserves identity, wardrobe, voice, prop, world, screen direction, and light state.
9. A deterministic collision test stays inside penetration/contact tolerances.
10. A protected camera cannot pass through collision geometry.
11. The 3D path renders an exact native 4K frame sequence.
12. The neural path produces a verified 4K delivery master while reporting true native generation resolution.
13. Audio stems, subtitles, loudness, and A/V sync pass QC.
14. Missing/failed optional packs fall back without deleting completed work.
15. The 15-image, 3-video, and 3-voice limits are enforced in UI, API, storage, and manifests.
16. Voice-reference generation is blocked without valid consent.
17. Local-only mode makes no outbound connection.
18. Installer, application update, model packs, and final provenance hashes verify.
19. A project created in the previous supported version migrates and can roll back from backup.
20. A redacted support bundle contains enough technical information to diagnose a failed job without exposing media or prompts.

## 4. Current repository gap analysis

The current `agent/local-cinematic-studio` branch is architecture/UI/packaging scaffolding, not a production-complete application.

Known blockers include:

- `local_studio.app` and `local_studio.jobs` import `local_studio.generation`, but that module is absent on the branch.
- The worker scaffold imports absent generation, connector, and story-memory modules.
- The desktop scaffold expects configuration fields/methods not present in the committed settings class.
- Generation is still designed around an API-process thread executor rather than the required crash-isolated worker process.
- Hardware status is a basic CUDA/FFmpeg snapshot; it lacks accelerator/driver/codec benchmarking and model eligibility planning.
- The request contract exposes only 480p/720p native rendering and lacks delivery resolution, world/character/shot IDs, dialogue tracks, physics state, and continuity packets.
- Project persistence is JSON/file oriented and lacks the SQLite WAL story/world/event schema.
- There is no OpenUSD world service, physics service, 3D asset validation service, finishing service, connector registry, model-pack manager, audio timeline, or 4K QC contract.
- The installer workflow has not passed a clean supported Windows machine install/run test.
- Real CUDA/model-weight paths have not been validated by the current branch’s mock-oriented tests.

These are implementation tasks. Documentation does not close them.

## 5. Delivery roadmap

### Phase 0 — Stabilise the current local studio

Deliverables:

- implement missing `generation`, `connectors`, `story_memory`, `finishing`, and worker-supervisor modules;
- align desktop/configuration contracts;
- move real generation out of the API process;
- add local token auth, cancellation, timeout, atomic output, and crash recovery;
- make browser and desktop launch tests pass;
- add CI for CPU/mock tests and a self-hosted real-GPU smoke path.

Gate: all existing tests plus worker-crash/resume and clean-launch smoke pass.

### Phase 1 — Persistent story, character audio, stitching, and 4K

Deliverables:

- SQLite WAL schemas and migrations;
- story/character/world/prop/scene/shot/take entities;
- voice profiles and consent;
- dialogue alignment, audio stems, lip-sync routing, subtitles;
- continuity packets and dependency invalidation;
- clip normalization, stitching, seam QC;
- verified 4K finish/export contract.

Gate: two-shot persistent story acceptance scenario.

### Phase 2 — Hardware planner and model packs

Deliverables:

- full hardware/codec probe and bounded microbenchmarks;
- hardware profile and capability planner;
- signed model manifests, isolated environments, resumable downloads, hashes, licences;
- OOM/fallback ladder, VRAM reservations, metrics;
- safe model enable/disable/repair UI.

Gate: hardware fixture matrix and model-pack install/rollback tests.

### Phase 3 — OpenUSD world, asset pipeline, and deterministic physics

Deliverables:

- USD stage/version service;
- Blender bridge and renderer;
- import/generated-asset validation and collision proxies;
- PhysX/Bullet/Warp adapters;
- physics editor, bake/cache, validation;
- camera/lens/lighting/colour system;
- native 4K 3D renders and control passes.

Gate: deterministic collision, camera collision, reopen, and native 4K tests.

### Phase 4 — World understanding and 3D-guided neural rendering

Deliverables:

- VLM scene interpretation;
- segmentation/tracking;
- camera/depth/point-map/3D reconstruction;
- Hunyuan3D and HY-World-class adapters;
- editable scene acceptance UI;
- control-pass-driven neural refinement and temporal QC.

Gate: multi-reference scene reconstructs into an editable world and produces repeatable multi-view shots.

### Phase 5 — Distributed/high-end capability

Deliverables:

- authenticated LAN workers;
- multi-GPU scheduling and shared artifact storage;
- Cosmos/HY-World high-end packs;
- render-farm queue and node maintenance;
- multi-user locking/audit where required.

Gate: interrupted remote job resumes without project corruption and preserves provenance.

## 6. Priority order

1. Make the current branch importable and crash-isolated.
2. Implement persistent story/audio/stitch/4K end to end.
3. Add truthful hardware/model eligibility.
4. Add USD/physics/3D authority.
5. Add world understanding and neural refinement.
6. Add high-end/distributed packs.

Do not begin high-end world-model integration before Phase 0 and Phase 1 release gates pass.

## 7. Definition of complete

The program is complete only when it can:

- inspect an unknown supported workstation and choose a safe pipeline;
- explain which models can and cannot run;
- build, load, edit, and version a persistent world;
- simulate, validate, and bake physical motion and collision;
- create cinematic cameras, lighting, and render/control passes;
- generate/refine neural video without losing world authority;
- give characters consented persistent voices and synchronized performances;
- preserve story/world/physics/audio continuity across shots and sessions;
- recover from worker, GPU, codec, connector, and application failures;
- stitch, mix, colour, verify, and export a real 4K master;
- reproduce and audit every source, model, version, setting, seed, licence, and decision.