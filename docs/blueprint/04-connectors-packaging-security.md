# Chapter 4 — Connectors, Local API, Packaging, Security, and Observability

## 1. Connector architecture

A connector is an optional capability provider. It never loads untrusted code into the control-plane process and it does not become the source of truth for project state.

### Connector handshake

```json
{
  "protocol_version": "1.0",
  "id": "comfyui-local",
  "version": "x.y.z",
  "transport": "http",
  "endpoint": "http://127.0.0.1:8188",
  "capabilities": ["video.generate", "image.upscale"],
  "health": "/system_stats",
  "auth": "local-token",
  "local_only": true,
  "limits": {"concurrency": 1}
}
```

Required operations:

```text
handshake()      capability/version/limits
estimate(job)    VRAM/RAM/disk/time estimate
submit(job)      connector job ID
status(job_id)   progress/events/log cursor
cancel(job_id)   acknowledgement
artifacts(job_id) typed artifact URIs and hashes
health()         dependency/runtime status
shutdown()       graceful stop
```

Every connector request has a project/job/node/trace ID, schema version, idempotency key, deadline, artifact allow-list, output quota, and trust level.

## 2. Planned connectors

### Production connectors

- **Blender:** headless and interactive bridge, USD import/export, render, simulation, asset validation, and preview.
- **PhysX/Bullet/Warp:** physics adapters selected by scene and hardware.
- **ComfyUI:** optional experimental model workflows behind a reviewed workflow adapter; community nodes never enter the core environment.
- **Local model servers:** llama.cpp, Ollama, vLLM/OpenAI-compatible endpoints, TTS, lip-sync, audio, and perception services.
- **Unreal/Unity:** optional world preview/render bridge with coordinate/material adapters.
- **LAN render worker:** authenticated workstation/render-node connector.
- **Storage:** local NAS or object store with content hashes and leases.
- **Optional cloud:** disabled by default and separately consented.

### Connector safety

- loopback-only default;
- remote endpoints require explicit LAN/cloud mode;
- TLS and mutual authentication for LAN workers;
- least-privilege process user;
- project-root path allow-list;
- no shell command strings—argument arrays only;
- timeouts, cancellation, CPU/RAM/VRAM/disk quotas;
- signed/hashed workflow definitions where possible;
- connector output is schema-validated and copied into the artifact store;
- connector failure cannot corrupt the main database or approved artifacts.

## 3. Local API

The control plane exposes a versioned API on `127.0.0.1` by default.

```text
GET  /v1/health
GET  /v1/hardware/profile
POST /v1/hardware/scan
GET  /v1/capabilities
GET  /v1/model-packs
POST /v1/model-packs/{id}/install
POST /v1/model-packs/{id}/verify
POST /v1/projects
GET  /v1/projects/{id}
POST /v1/worlds
GET  /v1/worlds/{id}
PATCH /v1/worlds/{id}
POST /v1/characters
PATCH /v1/characters/{id}
POST /v1/scenes
POST /v1/shots
POST /v1/takes
POST /v1/jobs
GET  /v1/jobs/{id}
POST /v1/jobs/{id}/cancel
POST /v1/jobs/{id}/resume
GET  /v1/jobs/{id}/events
POST /v1/exports
GET  /v1/connectors
POST /v1/connectors/{id}/test
GET  /v1/events
```

### API rules

- random per-launch desktop token;
- CSRF protection for browser requests;
- strict Pydantic/JSON-Schema validation;
- idempotency keys for writes;
- optimistic concurrency/version checks for story/world edits;
- range/streaming responses for media;
- no arbitrary filesystem paths from clients;
- errors have stable codes, user message, technical detail, and retryability.

## 4. Desktop supervisor

### Production target

Use a Tauri/Rust desktop supervisor and the operating system webview. Responsibilities:

- start/stop the control plane and workers;
- reserve a loopback port and token;
- heartbeat and crash monitoring;
- update/rollback;
- credential-store integration;
- file-open protocol and project association;
- support bundle and safe mode;
- signed model-pack/application manifests;
- worker process groups and cleanup.

The Python/pywebview launcher may remain a development route, but it is not the final process-isolation or update boundary.

## 5. One-click installer

### Core installer contents

- native supervisor and UI;
- control-plane runtime;
- SQLite/OpenUSD schemas and migrations;
- FFmpeg/ffprobe build with matching notices;
- minimal CPU-safe media and hardware-probe components;
- model/connector manager;
- documentation, licence notices, SBOM, and support tools.

Large model weights are not welded into the installer. They are optional, resumable, versioned packs with size estimates, hashes, licences, and offline bundle support.

### Windows first

The first supported release is a signed Windows installer EXE. It installs a maintainable on-disk application bundle rather than a self-extracting one-file Python executable. It provides repair, uninstall, Start Menu/Desktop shortcuts, dependency check, and safe-mode launch.

Linux packages and notarized macOS bundles follow after Windows clean-machine, update, and rollback gates pass.

### Updates

- application and model packs update independently;
- delta application updates where practical;
- A/B or previous-version rollback;
- database/schema backup before migration;
- signed release manifest and checksum verification;
- update cannot delete user projects or approved artifacts.

## 6. Model and environment installation

Each incompatible Python/CUDA stack has an isolated environment/runtime directory, preferably managed by `uv` or an equivalent locked pack installer.

Installation sequence:

1. show pack size, expanded size, hardware eligibility, licence, and known limitations;
2. obtain explicit acceptance;
3. download to a partial/resumable cache;
4. verify hashes;
5. build/install the isolated environment from a lockfile;
6. run dependency/driver checks;
7. launch worker health test;
8. run a tiny smoke inference/encode where feasible;
9. mark pack enabled only after success.

A failed installation remains disabled and removable without affecting the application.

## 7. Security and privacy

- no telemetry by default;
- no network model download without an explicit user action;
- local-only mode blocks non-loopback connectors;
- per-launch API token and same-site cookies;
- operating-system credential store for secrets;
- strict path validation beneath project/cache roots;
- streamed uploads with size/type/magic validation;
- archive-bomb, symlink, and path-traversal protections;
- least-privilege workers and connector allow-lists;
- signed updates and checksummed model packs;
- face and voice consent records with scope and revocation;
- output provenance and optional content credentials/watermarking;
- support bundles redact media, prompts, credentials, identity, and personal paths unless explicitly included.

## 8. Licence policy

Every source, model, connector, workflow, and asset records:

- licence identifier/text/reference;
- commercial-use and redistribution flags;
- attribution/notice requirements;
- derivative/output restrictions when known;
- source revision and hash;
- user acceptance timestamp.

The model router cannot enable a pack whose licence conflicts with the chosen distribution/use mode. AGPL/GPL connectors are kept out-of-process and distributed only after a licence review.

## 9. Observability

### Structured events

Every log/metric includes timestamp, severity, component, worker ID, project/job/node/trace ID, event code, retryability, and redaction state.

### Metrics

- CPU/RAM/disk/network;
- GPU utilisation, VRAM, temperature, power, and throttling;
- worker launch/health/exit history;
- node duration and cache hit rate;
- model load/inference/encode times;
- queue time and concurrency;
- connector latency/errors;
- encode/decode throughput;
- quality and repair attempts.

### Crash reports

The supervisor captures exit code, signal/exception, driver/runtime versions, last structured events, resource snapshot, and log tail. It does not capture source media or prompts unless diagnostic capture is explicitly enabled.

## 10. Support and recovery

- safe mode starts without model packs or third-party connectors;
- database integrity check and automated backup restore;
- orphaned partial-job cleanup after confirmation;
- model-pack repair/reverify;
- cache rebuild from artifact manifests;
- project export/import bundle;
- redacted support bundle;
- clear user-facing recovery action for OOM, driver, disk, codec, licence, and connector failures.