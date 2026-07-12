# Chapter 3 — Persistent Storytelling, Character Voice, Audio, Stitching, and 4K

## 1. Persistent narrative model

The story engine is an event-sourced state machine. It records what changed, when, why, who approved it, and which downstream artifacts depend on it.

### Project hierarchy

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

### Memory scopes

- **Project:** style bible, colour language, framing rules, forbidden content, delivery defaults.
- **Character:** identity, proportions, appearance references, wardrobe variants, voice, relationships, emotional arc, injuries/changes, active goals.
- **World:** geometry/version, locations, time, weather, lighting, damage, active objects, simulation state.
- **Scene:** dramatic objective, entrance/exit state, continuity requirements, selected world layer.
- **Shot:** blocking, camera, action beats, dialogue, active assets, terminal frame, terminal physics state.
- **Take:** exact models, settings, seeds, manifests, artifacts, QC, approval, and superseded state.

SQLite in WAL mode stores these records, events, indexes, approvals, and artifact references. Large media and tensors remain in the content-addressed artifact store.

## 2. Character bible

A character record includes:

- immutable character ID and version;
- name/aliases and narrative role;
- consented identity reference set;
- appearance invariants and editable attributes;
- body scale/proportions and rig/skeleton references;
- wardrobe, accessories, props, and variants;
- voice profile, languages, pronunciation dictionary, and consent scope;
- emotional baseline, relationships, current goals, and story state;
- accepted changes, injuries, dirt/wetness/damage, and their scene ranges;
- user locks and source/provenance.

A wardrobe or identity change creates a new version; it does not overwrite prior approved shots.

## 3. Continuity packet

Before a shot runs, the continuity compiler creates an immutable `ContinuityPacket` containing only relevant state:

- selected character and object identity references;
- accepted world/character/prop versions;
- prior terminal frame and optional overlap frames;
- previous transforms, velocities, contact state, and physics-cache ID;
- camera axis, screen direction, eyelines, lens, focus, and exposure;
- light rig, weather, time of day, wardrobe, and prop state;
- active voice profile, dialogue history, phoneme/viseme requirements;
- positive requirements, negative constraints, and locked properties;
- provenance and schema version.

The packet is stored with the take so the shot can be reproduced or audited.

## 4. Dependency invalidation

Changes invalidate only dependent nodes:

- dialogue edit → voice, alignment, performance, mix, subtitles, and final export;
- camera edit → camera-dependent render/control passes and downstream neural/finish nodes;
- physics/material edit → simulation cache, render/control passes, and downstream outputs;
- identity/wardrobe edit → affected shots after the version boundary;
- model version change → model-dependent nodes only;
- final encode setting → encode/QC/export only.

Approved upstream sources and unrelated shots remain valid.

## 5. Voice profiles and consent

The app supports up to three active voice/audio reference samples per generation. A character voice profile records:

- profile and character IDs;
- sample artifact hashes;
- owner/performer and consent record;
- permitted uses, project scope, language scope, expiry/revocation;
- model/engine, language, speaker embedding, and pronunciation dictionary;
- pitch/range, cadence, intensity, accent, and emotional controls;
- reference transcript and quality measurements.

No cloning, conversion, or reference-conditioned voice job runs without a valid consent record. Revocation prevents new generation while preserving audit history.

## 6. Dialogue and performance pipeline

```text
Script line
 → language/pronunciation normalization
 → TTS or voice generation
 → forced alignment
 → phoneme and viseme timeline
 → S2V or lip-sync/facial animation
 → dialogue stem
 → mix, subtitles, and QC
```

### Routing

- Wan S2V is used only when its model pack and hardware qualify.
- A high-quality lip-sync worker may use LatentSync-class models.
- A lightweight route may use MuseTalk-class models.
- 3D characters receive phoneme/viseme curves and facial animation before rendering.
- If a voice engine is absent, the app preserves recorded dialogue and standard audio editing; it never invents a fake “cloned” voice.

## 7. Audio timeline

Internal audio processing uses 48 kHz floating-point stems. The editor provides:

- dialogue and ADR;
- room tone and ambience;
- foley and sound effects;
- music;
- bus routing and master;
- fades and crossfades;
- clip gain, automation, EQ, compression, de-essing, reverb, delay;
- ducking and sidechain control;
- pan/spatial metadata;
- limiter, loudness presets, and true-peak checks;
- subtitles/captions and export stems.

Every generated or imported audio artifact records sample rate, channel layout, duration, source, model/engine, prompt, consent, and hash.

## 8. Video stitching

Stitching is a deterministic finishing operation, not a concatenation shortcut.

### Normalization before assembly

Each clip is normalized to a timeline contract:

- frame size and sample aspect ratio;
- constant frame rate and time base;
- pixel format, colour primaries, transfer, matrix, and range;
- codec/profile where stream-copy is not valid;
- audio sample rate, channel layout, loudness range, and padding;
- start timestamps and duration.

### Transitions

Supported transitions include hard cut, dissolve/crossfade, fade, and connector-provided optical-flow transitions. Audio uses matching cuts/crossfades with room-tone continuity.

The seam QC compares terminal/opening frames, identity state, world state, motion direction, camera axis, colour/exposure, audio waveform, and sync. It reports a seam score and recommends a cut, short dissolve, repair, or rerender.

### Story master

The story master references ordered approved takes. Replacing one take rebuilds only affected assembly/finish nodes.

## 9. 4K delivery contract

“4K” is a verified output state, not a marketing label.

### Paths

- **3D path:** native 3840×2160 UHD or 4096×2160 DCI frames from the renderer.
- **Wan path:** supported native generation → temporal video super-resolution → optional interpolation → controlled sharpening/grain → colour → encode.
- **LTX-2 or future pack:** native high-resolution audiovisual generation only when the local pack, hardware benchmark, and licence qualify.

Wan TI2V output must be labelled by its true native resolution. Upscaled output is “4K delivery,” not “native 4K.”

### Finishing stages

1. decode and technical validation;
2. optional deflicker/temporal repair;
3. temporal VSR or deterministic Lanczos fallback;
4. optional frame interpolation;
5. denoise/deblock/sharpen with bounded settings;
6. scene-referred colour transform and grade;
7. grain/dither after resize where selected;
8. audio mix and sync;
9. mezzanine/master encode;
10. ffprobe and QC verification;
11. checksum/provenance manifest.

Neural upscalers are replaceable workers. Their failure falls back without deleting the native render.

## 10. 4K verification

A master is complete only when automated QC confirms:

- exact requested dimensions and sample aspect ratio;
- expected frame rate, frame count, duration, and time base;
- no decode errors, truncated tail, missing frames, or unexpected black-frame tail;
- audio stream presence, sample rate, channel layout, duration, and A/V sync;
- codec/profile/level, bit depth, pixel format, GOP, and bitrate constraints;
- colour primaries, transfer, matrix, range, and mastering metadata where applicable;
- peak/loudness limits;
- checksum and provenance record.

Export presets include H.264/H.265/AV1 delivery, high-quality mezzanine, image sequences, WAV stems, subtitle files, project manifest, and optional USD package.

## 11. Quality control

### Semantic QC

- identity, wardrobe, prop, and world continuity;
- prompt/action completion;
- face/body/hand integrity;
- object permanence and background stability;
- camera and motion consistency;
- dialogue content and pronunciation.

### Technical QC

- flicker/strobing and duplicate/frozen frames;
- optical-flow discontinuity;
- clipping, loudness, silence, and sync;
- compression artifacts and illegal colour levels;
- render-pass/frame-count integrity;
- seam differences and transition validity;
- 4K delivery conformance.

### Physics QC

- penetration, missed contact, tunnelling, unstable energy, invalid body state, and camera collision.

QC results are stored with thresholds, measured values, affected time ranges, repair action, attempts, and approval/override state.

## 12. Story acceptance scenario

The minimum release scenario is a two-scene story with:

- one persistent character and voice;
- one persistent world and recurring prop;
- a physically simulated interaction;
- a camera/light change that preserves continuity;
- dialogue in both shots;
- a second shot conditioned on prior terminal world/character state;
- an approved stitched master;
- audio stems and subtitles;
- a verified 3840×2160 output;
- project close/reopen with identical story, world, voice, physics-cache, and approval state.