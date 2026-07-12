# Chapter 1 — Hardware, Runtime, Model Packs, and Scheduling

## 1. Hardware discovery

Hardware discovery runs on first launch, after a driver/runtime change, and on demand. It must finish without loading a generative model.

### Probe fields

**System:** OS/build, architecture, CPU model, physical/logical cores, instruction sets, RAM, swap/pagefile, power mode, and thermals where available.

**GPU:** vendor, model, device ID, total/free VRAM, driver, CUDA/ROCm/DirectML/Vulkan support, compute capability, BF16/FP16/FP8 support, peer-to-peer topology, and largest safe allocatable block.

**Media:** FFmpeg/ffprobe versions; H.264, HEVC, AV1, ProRes, NVENC, QSV, AMF, and VideoToolbox availability; 10-bit support; tested maximum encode/decode resolution.

**Storage:** free space, filesystem, cache locations, temp quota, and a bounded sequential read/write sample.

**Software:** application version, Python/runtime packs, OpenUSD, Blender, physics engines, model packs, connector versions, and incompatible drivers.

### Bounded microbenchmarks

Each test has a timeout and writes only disposable data:

- small FP16/BF16 matrix multiplication;
- 512 px VAE decode only when a relevant runtime is installed;
- 1080p and 4K encode/decode sample;
- disposable disk read/write sample;
- worker launch and IPC heartbeat;
- optional multi-GPU transfer test.

The result is cached under a fingerprint of hardware, drivers, runtime versions, and codecs.

## 2. Hardware profile and capability planning

The probe writes `hardware_profile.json` conforming to `docs/schemas/hardware_profile.schema.json`. The capability planner combines it with installed model manifests and produces:

- eligible model and engine packs;
- blocked packs with exact reasons;
- expected native resolution and finishing path;
- estimated peak VRAM, RAM, disk, and runtime;
- safe concurrency;
- required offload, tiling, quantisation, attention, and precision settings;
- Preview, Standard, Studio, and Max quality modes;
- fallback ladder and risk warnings.

Unsupported hardware must never be presented as a vague “slow mode.” The UI shows the missing requirement and a safe alternative.

## 3. Policy tiers

These are product policy tiers, not universal model promises. Eligibility is decided by a signed model manifest plus a local smoke benchmark.

| Tier | Typical hardware | Guaranteed capability |
| --- | --- | --- |
| CPU/edit | No supported GPU, 16–32 GB RAM | Story/project editing, ingest, proxies, transcoding, stitching, subtitles, audio editing, and low-cost planning. |
| Entry | 4–8 GB VRAM | Small perception/VLM packs, segmentation/depth, lightweight lip sync, 3D viewport, low-resolution previews; diffusion video only with an explicitly tested pack. |
| Creator | 12–16 GB VRAM | Stronger perception, eligible 3D asset generation, physics/3D preview, and optional distilled video packs after benchmark. |
| Studio 24 | 24 GB VRAM | Official Wan 2.2 TI2V-5B 720p baseline with offload as required, plus complete 4K finishing. |
| Studio 48 | 48 GB VRAM | Reduced offload, larger perception/reconstruction packs, and limited worker overlap. |
| Pro 80 | 80 GB VRAM or tested equivalent | Official Wan A14B/S2V class workloads, larger world models, and high-resolution simulation. |
| Enterprise | Multiple 80 GB-class GPUs | Distributed world generation/reasoning, render farm, and multi-user/LAN workers. |

## 4. Model-pack manifest

Every model pack has a signed `model_manifest.json` validated against `docs/schemas/model_manifest.schema.json`. It records:

- stable model ID and version;
- source repository/revision;
- code and weight licences, commercial-use flag, and required notices;
- artifact URLs, sizes, SHA-256 hashes, and disk expansion estimate;
- tasks, inputs, outputs, controls, supported durations/aspects/resolutions;
- supported operating systems, accelerators, precisions, and attention backends;
- minimum/recommended VRAM, RAM, disk, driver, and runtime versions;
- expected peak memory by mode;
- isolated environment pack, launch command, port, and health endpoint;
- safe concurrency, timeout, cancellation method, and fallback ladder;
- known limitations and quality warnings.

A pack cannot be enabled until its licence is accepted, hashes verify, health test passes, and hardware eligibility is proven.

## 5. Baseline model router

| Task | Baseline | Optional/high-end | Routing rule |
| --- | --- | --- | --- |
| Neural video | Wan 2.2 TI2V-5B | Wan A14B, LTX-2, connector models | Choose by hardware, controls, licence, native output, and benchmark. |
| Speech-driven video | lip-sync/performance adapter | Wan S2V-14B | S2V only on eligible high-memory hardware. |
| World reasoning | small/medium VLM | Cosmos Reasoner | Produces proposals and confidence, not physics truth. |
| Segmentation/tracking | SAM 2 adapter | specialist trackers | Stable IDs must survive frames and shots. |
| Depth/camera/3D | depth/VGGT adapter | HY-World/WorldMirror | Store uncertainty and allow correction. |
| 3D assets | validated import | Hunyuan3D/connector packs | Generated assets require scale, topology, material, and collision validation. |
| Voice | local TTS baseline | CosyVoice/other consented packs | Voice generation is blocked without consent when a reference is used. |
| Lip sync | lightweight adapter | LatentSync/high-quality adapter | Select by crop, duration, VRAM, and QC. |
| Audio generation | deterministic library/mixer | local video-to-audio pack | Optional and licence-gated. |

### Wan 2.2 contract

- TI2V-5B is the default supported neural renderer for the tested 24 GB CUDA tier.
- Its product contract is native 1280×704 or 704×1280 at 24 fps; it is not labelled native 4K.
- A14B and S2V packs are high-memory options.
- The app’s 15 image, 3 video, and 3 audio limits do not imply that one model natively conditions on all 21 files. The compiler routes references to the stages that can genuinely use them: VLM analysis, reconstruction, style extraction, motion analysis, direct model conditioning, world building, audio, or prompt constraints.

## 6. Runtime isolation

The final desktop target uses a small native supervisor. The API/control plane, each model pack, the 3D/physics engine, audio engine, and finishing engine run in separate processes.

Required worker interface:

```text
health() -> WorkerHealth
capabilities() -> CapabilitySet
estimate(request, hardware_profile) -> ResourceEstimate
submit(job_node) -> WorkerJobId
status(worker_job_id) -> ProgressEvent
cancel(worker_job_id) -> Acknowledgement
fetch_artifact(uri) -> Artifact
shutdown(graceful_timeout)
```

Large tensors and frames are transferred through artifact URIs, memory-mapped files, or shared memory—not JSON. Every message carries `project_id`, `job_id`, `node_id`, `trace_id`, `schema_version`, and an idempotency key.

A crash or native-extension fault may terminate only its worker. The supervisor releases locks, records the exit code/log tail, and offers a safe retry.

## 7. Resumable DAG scheduler

The production DAG is:

```text
INGEST → UNDERSTAND → PLAN → BUILD/LOAD WORLD → SIMULATE
→ RENDER CONTROL PASSES → GENERATE/REFINE VIDEO
→ GENERATE/ALIGN VOICE → PERFORMANCE/LIP SYNC
→ COMPOSITE → STITCH → UPSCALE/INTERPOLATE
→ MIX → COLOUR/ENCODE → QC → EXPORT
```

Every node has typed inputs/outputs, a content-derived cache key, resource estimate, timeout, retry policy, cancellation hook, logs, and atomic output validation.

### OOM/failure ladder

1. Restart a clean worker and release caches.
2. Enable supported CPU/model offload.
3. Enable tiled VAE/decode/upscale.
4. Lower precision only when the manifest permits it.
5. Reduce batch/concurrency.
6. Reduce preview resolution or temporal chunk size.
7. Route to another eligible worker/connector.
8. Fail clearly while preserving all successful upstream artifacts.

The scheduler must never silently reduce final delivery resolution, remove audio, disable physics, or change an approved asset/version.

## 8. Performance rules

- One diffusion job per GPU by default.
- Central VRAM reservation before worker launch.
- Lazy model loading and idle unload under memory pressure.
- Separate runtime/environment per incompatible CUDA/Python stack.
- Content-addressed deduplication for uploads, weights, and artifacts.
- Proxy media and thumbnails are generated once and reused.
- Cache/temp quotas use LRU cleanup; approved masters and source assets are never auto-deleted.
- Support bundles contain metrics and logs but redact prompts/media by default.