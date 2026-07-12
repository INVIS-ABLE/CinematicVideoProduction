# Cinematic Studio Local — Software and Engineering Blueprint

**Status:** architecture specification and delivery contract  
**Target release:** 1.0 local workstation edition  
**Repository:** `INVIS-ABLE/CinematicVideoProduction`  
**Primary operating mode:** private, local-first, hardware-scaled cinematic production  
**Reference limits:** 15 images, 3 video references, 3 voice/audio samples per generation

> This document defines the system that must be built. It does not claim that every item is already implemented. The present branch is a useful UI and packaging scaffold, but it must pass the implementation gates in §24 before it is called production-ready.

## 1. Product definition

Cinematic Studio Local is a modular application for creating short clips and persistent multi-shot stories from text, images, video, voice, and editable 3D worlds. It combines four distinct capabilities:

1. **AI video generation** for rapid visual ideation and final neural renders.
2. **Authoritative 3D world simulation** for repeatable geometry, cameras, lighting, collision, and physics.
3. **Persistent narrative memory** for characters, worlds, props, voices, scenes, shots, and continuity.
4. **Professional finishing** for dialogue, sound, stitching, colour, quality control, and verified 4K delivery.

AI-generated pixels are not treated as the source of truth for object geometry or physics. The canonical world is a versioned OpenUSD stage plus structured project state. Physics is computed by a deterministic simulation engine and baked before rendering. AI models may infer, propose, reconstruct, generate, or refine assets, but they cannot silently override authoritative world state.

## 2. Non-negotiable requirements

| Area | Requirement |
| --- | --- |
| Local-first | Runs offline after installation and model-pack download. No outbound network calls without explicit opt-in. |
| Hardware-scaled | Performs a fast hardware probe and exposes only eligible models and quality modes. |
| Stability | UI/control process survives model, GPU, Blender, connector, and codec crashes. |
| Modularity | Every heavy capability is a replaceable worker with a versioned manifest and health check. |
| World persistence | Geometry, transforms, materials, lights, cameras, physics state, and shot state persist between sessions. |
| Story persistence | Character identity, wardrobe, voice, relationships, scene state, dialogue, and continuity persist across clips. |
| Physics | Collision and dynamics are engine-computed, reproducible, inspectable, and bakeable. |
| 4K | Supports native 4K from the 3D path and verified 4K finishing from neural video paths. |
| Audio | Character voices, dialogue timing, lip sync, ambience, foley, music, mixing, and subtitles are first-class tracks. |
| References | Hard limits of 15 images, 3 videos, and 3 voice/audio samples are enforced in UI, API, and storage. |
| Resumability | Every expensive job is a checkpointed DAG; interrupted work resumes from the last valid node. |
| Provenance | Every output records source hashes, model versions, settings, seeds, licences, and consent metadata. |
| Packaging | Distributed as a one-click desktop installer; large model packs are optional, versioned downloads. |

## 3. System architecture

```text
┌──────────────────────────────── Desktop Supervisor ────────────────────────────────┐
│ Tauri/Rust shell • local auth token • updater • crash monitor • support bundle      │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │ loopback IPC
┌──────────────────────────────── Control Plane ──────────────────────────────────────┐
│ Project API • capability API • scheduler • DAG engine • event stream • policy       │
│ Story memory • model registry • connector registry • provenance • QC coordinator    │
└───────────┬───────────────────┬───────────────────┬───────────────────┬──────────────┘
            │                   │                   │                   │
     ┌──────▼──────┐     ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
     │ AI workers  │     │ World worker │    │ Audio worker│    │ Finish worker│
     │ Wan/LTX/VLM │     │ USD/Blender  │    │ TTS/lipsync │    │ FFmpeg/VSR   │
     │ one process │     │ PhysX/Bullet │    │ mixer/align │    │ colour/QC    │
     │ per pack    │     │ render passes│    │             │    │             │
     └──────┬──────┘     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
            └───────────────────┴──────────────────┴───────────────────┘
                                       │
┌──────────────────────────────── Data Plane ─────────────────────────────────────────┐
│ SQLite WAL metadata/event log • OpenUSD stages • content-addressed media/artifacts   │
│ model packs • render cache • simulation cache • logs • checksums • backups           │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Process boundaries

The desktop shell, API/control plane, GPU model workers, 3D/physics worker, audio worker, and finishing worker must be separate operating-system processes. A native extension or CUDA failure may terminate only its worker. The supervisor records the failure, releases resource locks, offers a safe retry, and keeps the project open.

No heavy diffusion model may be loaded in the API process. The default is one diffusion job per GPU. Parallelism is allowed only when the hardware planner proves sufficient headroom.

### 3.2 Communication

- Local JSON-RPC or gRPC for commands and capability negotiation.
- Server-sent events or WebSocket for progress, logs, previews, and cancellation.
- Large tensors and frames pass by file-backed shared memory, memory-mapped arrays, or artifact URIs—not JSON.
- Every request carries `project_id`, `job_id`, `trace_id`, `schema_version`, and an idempotency key.
- Every worker implements `health`, `capabilities`, `estimate`, `submit`, `status`, `cancel`, and `shutdown`.

## 4. Hardware discovery and capability planner

Hardware discovery runs on first launch, after a driver change, and on demand. It must finish without loading a generative model.

### 4.1 Probe fields

**System:** operating system/build, architecture, CPU model, physical/logical cores, instruction sets, RAM, swap/pagefile, thermal/power mode, and available disk.

**GPU:** vendor, model, device ID, total/free VRAM, driver, CUDA/ROCm/DirectML/Vulkan availability, compute capability, BF16/FP16/FP8 support, peer-to-peer topology, and largest safe allocatable block.

**Media:** FFmpeg/ffprobe versions; available H.264, HEVC, AV1, ProRes, NVENC, QSV, AMF, and VideoToolbox encoders/decoders; 10-bit support; maximum tested resolution.

**Storage:** free space, sequential read/write sample, filesystem, model-cache location, project-cache location, and temp quota.

**Software:** Python/runtime packs, OpenUSD, Blender, PhysX/Bullet, model-pack versions, connector versions, and incompatible drivers.

### 4.2 Microbenchmarks

The probe performs bounded tests with hard timeouts:

- small FP16/BF16 matrix multiply;
- 512 px VAE decode only when the relevant runtime is installed;
- 1080p and 4K encode/decode sample;
- 1 GiB sequential disk read/write using a disposable file;
- worker launch and IPC heartbeat;
- optional multi-GPU transfer test.

Results are cached by a fingerprint of hardware, drivers, and runtime versions.

### 4.3 Hardware profile

The probe produces `hardware_profile.json`, validated against `docs/schemas/hardware_profile.schema.json`. The planner combines that profile with each model manifest and returns:

- eligible model packs;
- blocked model packs with exact reasons;
- estimated peak VRAM/RAM/disk;
- safe concurrency;
- preview/standard/studio/max quality modes;
- required offload, tiling, quantisation, and attention backends;
- expected native resolution and required finishing stages.

### 4.4 Recommended tiers

These are policy tiers, not promises. Final eligibility is decided by manifest requirements plus a local smoke benchmark.

| Tier | Typical hardware | Guaranteed product capability |
| --- | --- | --- |
| CPU/edit | No supported GPU, 16–32 GB RAM | Project/story editing, ingest, transcoding, stitching, subtitles, low-cost planning; no local diffusion video guarantee. |
| Entry | 4–8 GB VRAM | Small VLM/perception packs, segmentation/depth, lightweight lip sync, low-resolution previews, 3D viewport; video models only when a tested pack explicitly supports the device. |
| Creator | 12–16 GB VRAM | Stronger perception, Hunyuan3D shape/texture where eligible, 3D preview renders, optional distilled/quantised video packs after benchmark. |
| Studio 24 | 24 GB VRAM | Official Wan 2.2 TI2V-5B 720p baseline with CPU/offload settings; full local finishing and 4K delivery. |
| Studio 48 | 48 GB VRAM | Reduced offload, larger understanding models, faster multi-stage workflows, limited worker overlap. |
| Pro 80 | 80 GB VRAM or equivalent multi-GPU | Official Wan A14B and S2V-14B class workloads, high-resolution simulation, larger world/perception packs. |
| Enterprise | Multiple 80 GB-class GPUs | Full HY-World/Cosmos-class packs, distributed rendering, multi-user/LAN workers. |

Unsupported hardware must never be disguised as “slow mode.” The UI shows a precise incompatibility reason and an alternative workflow.

## 5. Canonical data model

### 5.1 Storage layers

1. **SQLite in WAL mode** stores projects, entities, story state, jobs, events, manifests, consent, and artifact indexes.
2. **OpenUSD** stores authoritative world geometry, hierarchy, transforms, cameras, lights, materials, animation, and physics schemas.
3. **Content-addressed blob storage** stores source media, generated media, meshes, textures, audio, caches, and exports by SHA-256.
4. **Optional vector index** supports semantic search, but never replaces relational/world state.

Large video, audio, meshes, tensors, or model weights must not be embedded in SQLite.

### 5.2 Project hierarchy

```text
Project
 ├─ Story
 │   ├─ Sequence
 │   │   ├─ Scene
 │   │   │   ├─ Shot
 │   │   │   │   └─ Take
 ├─ Character bible
 ├─ World bible
 ├─ Prop/asset library
 ├─ Voice profiles
 ├─ OpenUSD stages and variants
 ├─ Render/simulation DAGs
 └─ Exports and provenance
```

### 5.3 World state

A world has immutable IDs for entities and versioned snapshots. Each entity may contain:

- semantic label and confidence;
- parent/child relationship;
- transform, bounds, visibility, and lifecycle;
- mesh/3DGS/point-cloud references;
- PBR material and texture references;
- collision shape and collision layer;
- rigid/soft/cloth/fluid/body configuration;
- mass, density, friction, restitution, damping, gravity, and constraints;
- animation/skeleton state;
- lighting interaction flags;
- ownership/provenance and licence.

The canonical stage explicitly sets units, up-axis, colour workflow, timeline rate, and simulation rate. Import/export adapters perform coordinate conversion for Blender, Unreal, Unity, and other engines.

### 5.4 Story and continuity state

A character profile stores identity references, appearance invariants, body proportions, wardrobe variants, voice profile, languages, emotional baseline, relationships, current goals, injuries/changes, and consent scope.

Every shot snapshot stores:

- world and character version IDs;
- actor and prop transforms;
- camera transform, lens, sensor, focus, aperture, shutter, exposure, and white balance;
- lighting rig and environment state;
- physics cache version and terminal body states;
- opening/terminal frames and control passes;
- prompt plan, negative constraints, seed, model/runtime versions;
- dialogue, phonemes/visemes, timing, and audio tracks;
- continuity rules: screen direction, eyelines, axis, wardrobe, time, weather, and prop state.

A dependency graph invalidates only downstream nodes when a source asset, line of dialogue, world object, or model setting changes.

## 6. Modular subsystem specification

| Module | Responsibility | Required interface |
| --- | --- | --- |
| `supervisor` | Launch, auth, updates, worker lifecycle, crash recovery | process control, heartbeat, signed update |
| `hardware_probe` | Detect/benchmark hardware and codecs | `scan() -> HardwareProfile` |
| `capability_planner` | Select safe model/workflow settings | `plan(request, profile, manifests)` |
| `model_registry` | Install, verify, enable, disable model packs | manifest, hash, licence, health |
| `project_store` | SQLite event/state store and migrations | transactions, snapshots, backup |
| `artifact_store` | SHA-256 media/mesh/audio store | put/get/link/gc |
| `story_memory` | Character/world/scene/shot continuity | snapshot, query, invalidate |
| `director` | Script breakdown and shot planning | StorySpec → ShotSpec DAG |
| `world_understanding` | Detect/segment/track/reconstruct scene | references → scene proposal |
| `world_builder` | Convert proposals/assets to USD | scene proposal → USD layer |
| `asset_factory` | Create/import/validate 3D assets | image/text → validated asset |
| `physics` | Collision, dynamics, bake, validation | USD state → simulation cache |
| `cinematography` | Camera, lens, lighting, blocking, coverage | ShotSpec → camera/light plan |
| `renderer_3d` | Beauty and control-pass rendering | USD + shot → frames/passes |
| `renderer_ai` | Wan/LTX/other neural video generation | conditioning → clip |
| `voice` | TTS/voice conversion, consent, alignment | dialogue + profile → stems |
| `performance` | S2V/lip-sync/facial animation | video/image + audio → performance |
| `sound` | Foley/ambience/music/spatial mix | shot events → stems/mix |
| `finishing` | VSR, interpolation, colour, grain, encode | clips/stems → master |
| `quality_control` | Technical and semantic validation | artifact → QC report |
| `connectors` | ComfyUI, Blender, Unreal, LAN/cloud workers | common connector protocol |
| `job_engine` | Resumable DAG execution and caching | submit/resume/cancel/retry |

Each module is versioned and may be replaced without changing project schemas.

## 7. Model and engine routing

### 7.1 Baseline model roles

| Task | Baseline | Optional/high-end | Rule |
| --- | --- | --- | --- |
| Video generation | Wan 2.2 TI2V-5B | Wan A14B, LTX-2, connector models | Route by hardware, licence, controls, and benchmark. |
| Speech-driven character video | Wan 2.2 S2V-14B | Animate, lip-sync pipeline | S2V only when hardware and dependencies qualify. |
| World understanding | Qwen3-VL small/medium | Cosmos Reasoner | Produces proposals/confidence, not physics truth. |
| Segmentation/tracking | SAM 2 adapter | specialist trackers | Preserve stable object IDs across frames. |
| Camera/depth/3D | VGGT/depth adapter | WorldMirror/HY-World | Store confidence and allow correction. |
| 3D asset generation | Hunyuan3D | external/connector packs | Validate topology, scale, material, and collision proxy. |
| Physics | PhysX | Bullet CPU fallback, Warp custom simulation | Physics engine is authoritative. |
| Voice | CosyVoice adapter | external local TTS/VC packs | Consent and licence required. |
| Lip sync | MuseTalk low-resource | LatentSync higher-quality | Chosen by face crop, VRAM, and QC. |
| Video-to-audio | deterministic library/mixer | MMAudio-style adapter | Optional; licence-gated. |
| Workflow connector | Native workers | ComfyUI | Connector is not source-of-truth for project state. |

### 7.2 Wan 2.2 contract

- TI2V-5B is the default supported neural renderer for a 24 GB CUDA tier.
- Its native production contract is 1280×704 or 704×1280 at 24 fps, not native 4K.
- A14B T2V/I2V and S2V packs are high-memory options.
- Reference limits in the product do not imply native joint conditioning of all 21 files. The compiler assigns roles and routes each reference to the model or stage that can genuinely use it.
- Unsupported references remain in the project and may influence VLM analysis, 3D reconstruction, prompt constraints, style extraction, motion analysis, or audio—not silently discarded.

### 7.3 Model manifest

Every pack has a signed `model_manifest.json` containing:

- model ID, version, source revision, files, SHA-256 hashes;
- tasks, inputs, outputs, controls, supported aspect ratios/durations;
- code and weight licences plus commercial-use flags;
- supported OS/accelerators/precisions;
- minimum and recommended VRAM, RAM, disk, and drivers;
- environment pack, launch command, ports, health test;
- expected peak memory by mode;
- safe concurrency and fallback ladder;
- provenance fields and known limitations.

The application cannot install or enable a model with an unaccepted licence, failed hash, failed health test, or unsupported hardware.

## 8. World-understanding pipeline

```text
References
  → secure ingest / hash / metadata / consent
  → VLM scene interpretation
  → segmentation + stable tracking
  → camera, depth, point map and motion estimation
  → object/relationship graph
  → geometry reconstruction or asset generation
  → confidence report and user correction
  → OpenUSD stage
```

The VLM identifies objects, characters, actions, spatial relationships, materials, lighting clues, and uncertainty. Segmentation gives masks and stable IDs. Reconstruction estimates cameras, depth, point maps, normals, and tracks. The system then proposes an editable scene graph.

Every inferred property carries `source`, `confidence`, and `locked_by_user`. Low-confidence scale, occlusion, object permanence, or hidden geometry is surfaced for correction. Once accepted, the USD stage—not the VLM response—becomes authoritative.

## 9. 3D asset pipeline

Imported or generated assets pass through:

1. malware-safe import and format validation;
2. unit and coordinate normalization;
3. topology inspection and optional repair;
4. watertight/manifold check where relevant;
5. UV and PBR material validation;
6. scale/origin/pivot correction;
7. skeleton/animation validation;
8. level-of-detail generation;
9. collision proxy generation;
10. preview render and provenance record;
11. conversion to USD references/payloads.

Generated geometry is never admitted directly into a physics scene without a collision proxy and scale validation.

## 10. Physics and collision

### 10.1 Engine policy

- **PhysX/ovphysx** is the primary USD-aware physics backend on supported systems.
- **Bullet** is the cross-platform CPU fallback for rigid-body/collision workloads.
- **Warp** is an optional backend for custom GPU simulation, particles, geometry kernels, or differentiable workflows.
- Blender may host simulation and baking, but the project state remains in USD and the app database.

### 10.2 Simulation contract

- fixed simulation timestep, default 120 Hz for 24 fps output;
- explicit substeps and solver iterations;
- deterministic seeds;
- collision layers/masks and pair filters;
- continuous collision detection for fast bodies;
- contact offsets and penetration tolerances;
- material friction/restitution and mass/density rules;
- constraints/joints with limits;
- sleep/wake state and terminal velocities;
- cache/bake hash includes engine version, hardware mode, scene state, and settings.

Cross-hardware bitwise determinism is not assumed. The accepted simulation is baked to an immutable cache, and all renders use that cache.

### 10.3 Validation

The QC worker reports interpenetration depth, missed contacts, unstable energy, tunnelling, exploding constraints, invalid mass/scale, and camera collision. A shot fails the physics gate when configured tolerances are exceeded.

## 11. Lighting, camera, and cinematic visual system

### 11.1 Colour and light

- linear scene-referred rendering;
- OpenColorIO with ACES-compatible working and display transforms;
- physical light units, IES profiles, HDRI environments, emissive practicals;
- sun/sky, volumetrics, fog, bounce, reflection probes, and shadow controls;
- light linking and per-character key/fill/rim rigs;
- time-of-day and weather state stored in the world bible.

### 11.2 Camera model

Each camera stores focal length, sensor size, aperture/T-stop, focus distance, shutter angle, ISO/exposure, white balance, anamorphic squeeze, distortion, depth of field, motion blur, and optional rolling shutter. Camera paths use collision checks, occlusion checks, easing, speed/acceleration limits, and safe framing.

### 11.3 Cinematic planner

The director proposes blocking and coverage while enforcing:

- 180-degree axis and screen direction;
- eyelines and gaze targets;
- character/prop continuity;
- lens and depth-of-field continuity;
- shot-size progression and visual rhythm;
- motivated lighting and practical sources;
- safe camera paths and subject visibility;
- beginning/middle/end action within each clip.

Users can lock any camera/light/continuity property. Locked values override AI suggestions.

## 12. Rendering modes

### 12.1 Neural quick render

Text/image/reference analysis → Wan/LTX worker → optional audio/performance → finish/QC.

Use for ideation, atmospheric shots, or content where exact geometry is not required.

### 12.2 Authoritative 3D render

USD world → physics bake → Blender/Cycles/Eevee or engine connector → native 4K frames → audio/finish/QC.

Use when collision, repeatability, exact camera motion, product geometry, or persistent worlds matter.

### 12.3 3D-guided neural render

USD/physics → render depth, normals, albedo, segmentation, optical flow, motion vectors, object IDs, shadows, and rough beauty → supported control adapter/neural renderer → temporal consistency pass → finish/QC.

This is the preferred cinematic hybrid: world motion and camera are authoritative, while the neural stage adds photoreal detail or style.

### 12.4 Character performance

Character image/3D render + voice + motion/pose → Wan S2V when eligible, otherwise face/performance animation plus lip sync → composite → finish/QC.

## 13. Persistent storytelling

The story engine is an event-sourced state machine. It records what changed, when, why, and which artifact depends on the change.

### 13.1 Memory scopes

- **Global project:** style bible, colour language, framing conventions, prohibited content.
- **Character:** identity, voice, wardrobe, emotional arc, relationships, physical state.
- **World:** geometry, locations, time, weather, lighting, active objects, damage.
- **Scene:** dramatic objective, entrance/exit state, continuity constraints.
- **Shot:** blocking, camera, dialogue, action beats, terminal frame and terminal physics state.
- **Take:** exact settings, seed, model versions, artifacts, QC and approval.

### 13.2 Continuity compiler

Before a shot runs, it compiles a `ContinuityPacket` containing only relevant state:

- identity/reference embeddings or images;
- accepted world/character/prop versions;
- prior terminal frame and optional overlap frames;
- prior body/prop transforms and velocities;
- camera axis, eyelines, light state, weather, wardrobe;
- active voice profile and dialogue history;
- positive requirements and negative constraints.

The packet is immutable and stored with the take, making rerenders auditable.

## 14. Voice, dialogue, and sound

### 14.1 Voice profiles

A profile links a character to one of the three active voice samples and records owner, consent, permitted uses, languages, pronunciation dictionary, pitch/range, cadence, emotional controls, and expiry/revocation state.

No voice cloning or conversion job runs without an accepted consent record. Revocation prevents new generation while preserving project audit history.

### 14.2 Dialogue pipeline

```text
Script
 → pronunciation and language normalization
 → TTS/voice generation
 → forced alignment
 → phoneme/viseme track
 → S2V or lip-sync/facial animation
 → dialogue stem
 → mix and QC
```

### 14.3 Audio timeline

The editor supports dialogue, ADR, ambience, room tone, foley, effects, music, and master buses. Internal processing uses 48 kHz floating-point audio. It provides fades, crossfades, ducking, EQ, compression, de-essing, reverb, panning/spatial metadata, limiter, loudness presets, subtitles, and export stems.

Audio generation is optional and connector-based; deterministic libraries and recorded assets remain available when generative audio is absent or licence-restricted.

## 15. 4K delivery contract

“4K” is a verified output state, not a UI label.

### 15.1 Paths

- **3D path:** render natively at 3840×2160 UHD or 4096×2160 DCI.
- **Wan path:** generate at supported native resolution, then temporal video super-resolution, optional interpolation, controlled sharpening/grain, colour transform, and final encode.
- **LTX-2 pack:** may provide a native 4K audiovisual path when its local pack, hardware benchmark, and licence are accepted.

### 15.2 Verification

A master is complete only when ffprobe/QC confirms:

- exact requested dimensions and sample aspect ratio;
- expected frame rate, frame count, and duration;
- audio stream presence, sample rate, channel layout, and sync;
- selected codec/profile/bit depth/pixel format;
- colour primaries, transfer, matrix, and range metadata;
- no decode errors, truncation, black-frame tail, or missing frames;
- checksum and provenance manifest.

Default exports: UHD H.264/H.265, high-quality mezzanine, image sequence, WAV stems, subtitles, project manifest, and optional USD package.

## 16. Resumable job DAG

```text
INGEST
 → UNDERSTAND
 → PLAN
 → BUILD/LOAD WORLD
 → SIMULATE
 → RENDER CONTROL PASSES
 → GENERATE/REFINE VIDEO
 → GENERATE/ALIGN VOICE
 → PERFORMANCE/LIP SYNC
 → COMPOSITE
 → STITCH
 → UPSCALE/INTERPOLATE
 → MIX
 → COLOUR/ENCODE
 → QC
 → EXPORT
```

Every node has typed inputs/outputs, a content-derived cache key, resource estimate, timeout, retry policy, progress, logs, and cancellation hook. Node outputs are written atomically and marked valid only after checksum and schema validation.

### 16.1 OOM/failure ladder

For eligible jobs the scheduler may retry in this order:

1. release caches and restart a clean worker;
2. enable model/encoder offload;
3. enable tiled VAE/decode/upscale;
4. lower precision only when the manifest permits;
5. reduce batch/concurrency;
6. reduce preview resolution or temporal chunk size;
7. route to a different eligible worker/connector;
8. fail clearly without deleting successful upstream artifacts.

The scheduler must never silently lower final delivery resolution or remove audio.

## 17. Connector architecture

A connector is an optional capability provider, not a privileged plugin inside the main process.

### 17.1 Connector contract

```json
{
  "protocol_version": "1.0",
  "id": "comfyui-local",
  "transport": "http",
  "endpoint": "http://127.0.0.1:8188",
  "capabilities": ["video.generate", "image.upscale"],
  "health": "/system_stats",
  "auth": "local-token",
  "local_only": true
}
```

Required operations: capability handshake, estimate, submit, status/events, cancel, fetch artifact, health, and version.

### 17.2 Planned connectors

- ComfyUI for experimental node workflows;
- Blender headless/interactive bridge;
- Unreal/Unity bridge for world preview and rendering;
- local LLM/VLM servers such as llama.cpp, Ollama, or vLLM-compatible endpoints;
- LAN render worker with mutual authentication;
- NAS/object storage;
- optional cloud providers, disabled by default.

Connectors run with least privilege, path allow-lists, timeouts, output quotas, and an explicit trust level. Community nodes never load into the core application environment.

## 18. Local API

Versioned endpoints:

```text
GET  /v1/hardware/profile
POST /v1/hardware/scan
GET  /v1/capabilities
GET  /v1/model-packs
POST /v1/model-packs/{id}/install
POST /v1/projects
GET  /v1/projects/{id}
POST /v1/worlds
PATCH /v1/worlds/{id}
POST /v1/characters
POST /v1/scenes
POST /v1/shots
POST /v1/jobs
GET  /v1/jobs/{id}
POST /v1/jobs/{id}/cancel
POST /v1/jobs/{id}/resume
POST /v1/exports
GET  /v1/connectors
POST /v1/connectors/{id}/test
GET  /v1/events
```

The API binds to `127.0.0.1` by default and requires a random per-launch desktop token. LAN access is a separate opt-in mode with TLS and authentication.

## 19. Performance and anti-bloat design

- The installer contains the supervisor, UI, control plane, FFmpeg tooling, schemas, and minimal CPU-safe runtime—not every model.
- Model packs are separate resumable downloads with hashes and disk estimates.
- Each incompatible Python/CUDA stack has an isolated `uv` environment or container-like runtime directory.
- Model workers load lazily and unload under memory pressure.
- A central VRAM reservation service prevents overlapping jobs from overcommitting a GPU.
- Preview proxies, thumbnails, audio waveforms, and render caches are generated once and reused.
- Content-addressed deduplication prevents duplicate uploads/models.
- Temp/cache quotas and LRU cleanup protect disk space; approved masters and sources are never auto-deleted.
- FFmpeg normalizes clip codecs, frame rates, timebases, audio layouts, and colour metadata before stitching.
- The UI virtualizes long lists/timelines and never decodes full-resolution media on the main thread.

## 20. Security, privacy, licensing, and provenance

- loopback-only default;
- no telemetry or model download without consent;
- random session token and CSRF protection;
- strict project-root path validation;
- streamed uploads with size/type checks and malware-safe handling;
- signed application updates and checksummed model packs;
- encrypted secrets using the operating-system credential store;
- licence registry for code, weights, outputs, and commercial restrictions;
- face/voice consent records and scope enforcement;
- provenance manifest and optional content credentials/watermarking;
- support bundles redact media, prompts, tokens, and personal paths by default.

## 21. Desktop packaging and updates

### 21.1 Production target

Use a Tauri/Rust supervisor and native webview for a small, secure desktop shell. The current Python/pywebview launcher may remain an interim development route, but it is not the final reliability boundary.

### 21.2 Installer

- one Windows installer EXE as the first supported target;
- on-disk application bundle rather than a self-extracting single-file Python executable;
- Start Menu/desktop shortcuts, uninstaller, repair mode;
- prerequisite and driver check;
- optional offline model-pack bundles;
- delta application updates with rollback;
- model packs updated independently from the application;
- release manifest, signatures, SBOM, third-party notices, and hashes.

Linux packages and macOS notarized bundles follow after the Windows release gates pass.

## 22. Observability and support

- structured JSON logs with trace/job/node IDs;
- worker stdout/stderr capture and crash codes;
- GPU VRAM/utilisation/temperature/power sampling;
- CPU/RAM/disk/codec metrics;
- node timings and cache hit rates;
- model launch and health history;
- user-readable failure reason plus technical report;
- exportable, redacted support bundle;
- no media or prompt contents in logs unless the user explicitly enables diagnostic capture.

## 23. Testing and release gates

### 23.1 Test layers

- schema and migration unit tests;
- storage atomicity and path-security tests;
- hardware-profile fixtures for NVIDIA/AMD/Intel/CPU;
- worker crash/OOM/timeout chaos tests;
- model manifest and licence tests;
- USD round-trip and coordinate conversion tests;
- deterministic physics scenes;
- golden camera/light/control-pass tests;
- audio alignment, clipping, loudness, and sync tests;
- stitching/timebase tests;
- 4K codec/probe tests;
- installer clean-machine tests;
- upgrade/rollback and project-migration tests.

### 23.2 Acceptance criteria

A 1.0 release must prove all of the following:

1. Fresh install launches offline on a clean supported machine.
2. Hardware scan completes without loading a generative model and reports exact eligibility reasons.
3. The app does not offer Wan TI2V when less than the tested requirement is available.
4. Killing a GPU worker does not terminate the UI or corrupt the project.
5. A job resumes after application restart from the last valid DAG node.
6. Reopening a project restores characters, voices, worlds, shots, physics caches, and approvals.
7. A two-shot story preserves character/wardrobe/voice/world state and produces a stitched master.
8. A deterministic collision test stays within configured penetration/contact tolerances.
9. A 3D camera path cannot pass through protected collision geometry.
10. A 3840×2160 master passes codec, frame, duration, audio-sync, and colour-metadata probes.
11. Missing/failed optional model packs fall back without losing completed work.
12. Reference limits are enforced at 15 images, 3 videos, and 3 voice samples in all layers.
13. Voice generation is blocked without valid consent.
14. No outbound connection occurs in local-only mode.
15. Installer, model packs, and output provenance hashes verify.

## 24. Current repository gap analysis

The current branch must not be represented as production-ready until these blockers are closed:

- `local_studio.app` and `local_studio.jobs` import `local_studio.generation`, but that module is absent.
- The worker scaffold imports absent `generation`, `connectors`, and `story_memory` modules.
- The desktop scaffold calls configuration methods/fields that do not yet exist.
- Heavy generation is still designed around a thread executor in the API process rather than the required crash-isolated process worker.
- Hardware status is a basic CUDA/FFmpeg check; it lacks codec, driver, accelerator, disk, benchmark, and eligibility planning.
- The request contract exposes only 480p/720p native resolution and lacks delivery resolution, world, character, dialogue, physics, and continuity IDs.
- Project persistence is file/JSON oriented and lacks the SQLite WAL story/world/event schema.
- There is no OpenUSD world service, physics service, finishing service, connector registry, model-pack manager, 4K QC contract, or audio timeline.
- The installer workflow has not passed a clean Windows machine build/run test.

These are implementation tasks, not documentation-only items.

## 25. Delivery roadmap

### Phase 0 — Stabilise the local studio

Implement missing modules and configuration contracts, move generation to a subprocess worker, add auth, repair tests, and validate browser/desktop launch. Gate: all existing studio tests plus crash-recovery smoke pass.

### Phase 1 — Persistent story, audio, stitching, and 4K finishing

Add SQLite WAL, character/world/shot schemas, voice profiles/consent, dialogue alignment, audio timeline, continuity packets, clip normalization/stitching, and verified 4K export. Gate: two-shot persistent story acceptance test.

### Phase 2 — Hardware planner and model packs

Implement full probe, manifests, isolated environments, download/hash/licence manager, eligibility matrix, OOM ladder, and metrics. Gate: hardware fixture matrix and clean install.

### Phase 3 — OpenUSD world and deterministic physics

Add USD stage service, asset import/validation, Blender bridge, PhysX/Bullet adapter, collision editor, simulation bake, lighting/camera system, and native 4K 3D render. Gate: deterministic collision/camera tests.

### Phase 4 — World understanding and 3D-guided neural rendering

Add VLM, segmentation/tracking, camera/depth/3D reconstruction, Hunyuan3D/HY-World adapters, control-pass renderer, and neural refinement. Gate: editable reconstructed scene and repeatable multi-view shot.

### Phase 5 — Distributed and enterprise capability

Add authenticated LAN workers, multi-GPU scheduling, Cosmos/HY-World high-end packs, shared artifact storage, and render-farm controls. Gate: interrupted remote job resumes without project corruption.

## 26. Reference implementation stack

- Wan 2.2: https://github.com/Wan-Video/Wan2.2
- OpenUSD: https://openusd.org/
- NVIDIA PhysX: https://github.com/NVIDIA-Omniverse/PhysX
- NVIDIA Warp: https://github.com/NVIDIA/warp
- NVIDIA Cosmos: https://github.com/NVIDIA/Cosmos
- HY-World 2.0: https://github.com/Tencent-Hunyuan/HY-World-2.0
- Hunyuan3D 2: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- LTX-Video/LTX-2: https://github.com/Lightricks/LTX-Video
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- FFmpeg: https://ffmpeg.org/
- Tauri: https://tauri.app/
- uv: https://github.com/astral-sh/uv

## 27. Definition of “complete”

The application is complete only when it can inspect an unknown supported workstation, choose a safe pipeline, build or load a persistent world, simulate and validate physical motion, render or generate a shot, give characters consented voices, preserve continuity across shots, recover from worker failure, stitch and mix the story, verify a true 4K master, reopen the entire production later, and explain every model, source, setting, and decision used.
