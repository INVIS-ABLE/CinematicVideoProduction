# Chapter 2 — World Understanding, OpenUSD, 3D, Physics, Lighting, and Rendering

## 1. Authoritative world model

The application uses a versioned OpenUSD stage as the canonical scene. AI-generated pixels, VLM text, masks, point maps, and reconstructed meshes are proposals or artifacts; they are not authoritative until accepted into the stage.

The stage explicitly records:

- units and up-axis;
- timeline rate and simulation rate;
- hierarchy, references, payloads, layers, variants, and overrides;
- geometry, transforms, visibility, bounds, and lifecycle;
- PBR materials, textures, colour spaces, and provenance;
- cameras, lights, environments, and render settings;
- animation, skeletons, constraints, physics schemas, and baked caches.

SQLite stores project/story metadata, events, approvals, indexes, and artifact URIs. Large media, meshes, textures, tensors, simulations, and model weights remain outside SQLite in a content-addressed store.

## 2. World-state entity contract

Every entity has an immutable ID and versioned state. The minimum entity record includes:

- semantic class/name and confidence;
- source/provenance and licence;
- parent/child relationship;
- transform, pivot, local/world bounds, visibility, and active interval;
- mesh, 3D Gaussian, point-cloud, texture, and material references;
- collision proxy and collision layer/mask;
- rigid, soft, cloth, fluid, particle, or kinematic configuration;
- mass/density, friction, restitution, damping, gravity, and constraints;
- skeleton/animation state;
- user locks and accepted/rejected inferred properties.

Every inferred field carries `source`, `confidence`, and `locked_by_user`. Low-confidence geometry, hidden surfaces, scale, occlusion, object permanence, or material inference must be shown for correction.

## 3. World-understanding pipeline

```text
15 images + up to 3 videos
  → secure ingest, hashes, metadata, consent
  → VLM scene interpretation
  → segmentation and stable tracking
  → camera, depth, point map, normals, and motion estimation
  → object/relationship graph
  → asset matching, reconstruction, or generation proposal
  → confidence report and user corrections
  → accepted OpenUSD layer
```

### VLM responsibilities

The VLM proposes objects, people, actions, spatial relationships, materials, lighting cues, scene purpose, and uncertainties. It may also extract a shot list or continuity constraints. It must not author irreversible world state without validation.

### Segmentation/tracking

A segmentation/tracking adapter maintains stable object IDs across frames, reference videos, and shots. Masks become control passes, matte inputs, visibility evidence, and reconstruction constraints.

### Camera/depth/reconstruction

Depth/camera adapters estimate intrinsics, extrinsics, depth, point maps, normals, optical flow, and tracks. Multi-view inputs are preferred. The system stores confidence and reprojection error and allows manual scale/camera correction before world acceptance.

High-end world packs such as HY-World/WorldMirror or Cosmos-class models are optional workers. They may create or reason about large environments, but their outputs still pass through the same validation and USD acceptance path.

## 4. 3D asset pipeline

Imported or generated assets pass these gates:

1. malware-safe import and file-format validation;
2. coordinate, unit, scale, origin, and pivot normalization;
3. topology inspection and optional repair;
4. watertight/manifold check where required;
5. UV, texture, colour-space, and PBR material validation;
6. skeleton, skinning, morph-target, and animation validation;
7. level-of-detail and proxy generation;
8. collision proxy generation;
9. preview turntable and scale/contact test;
10. provenance/licence record;
11. conversion to USD reference/payload.

A generated mesh cannot enter a physics scene until scale and collision validation pass. Render geometry and collision geometry are separate assets.

### Recommended asset workers

- validated user/imported assets first;
- Hunyuan3D-class packs for image/text-to-3D proposals;
- procedural Blender/geometry-node assets;
- optional marketplace/DCC connectors with explicit licences;
- HY-World-class scene generation for high-end hardware.

## 5. Physics architecture

### Engine policy

- **PhysX/ovphysx** is the primary USD-aware backend on supported systems.
- **Bullet** is the cross-platform CPU fallback for rigid-body/collision workflows.
- **Warp** is an optional GPU kernel/simulation backend for particles, geometry, custom constraints, or differentiable workflows.
- Blender may host simulation and baking, while the authoritative scene/version remains in USD and the project database.

### Simulation contract

- fixed timestep, default 120 Hz for 24 fps output;
- explicit substeps and solver iterations;
- deterministic seeds;
- collision layers/masks and pair filters;
- continuous collision detection for fast objects;
- contact offsets, penetration tolerance, friction/restitution combine rules;
- mass/density and inertia validation;
- joints/constraints with limits and break thresholds;
- sleep/wake state, terminal transforms, velocities, and angular velocities;
- cache hash includes scene version, engine/runtime version, settings, and hardware mode.

Cross-hardware bitwise determinism is not assumed. Once a simulation is approved, it is baked to an immutable cache and every render uses that cache.

### Physics validation

The QC worker reports:

- interpenetration depth and duration;
- missed or unstable contact;
- tunnelling;
- energy explosions or non-finite values;
- invalid scale, mass, inertia, or collision proxy;
- constraint instability;
- unsupported body/material combinations;
- camera-body collision and subject clipping.

A shot fails its physics gate when configured tolerances are exceeded. AI prompts cannot waive a failed physical validation without an explicit user override recorded in provenance.

## 6. Camera system

Every camera stores:

- transform and parent/rig;
- focal length and sensor dimensions;
- aperture/T-stop and focus distance;
- shutter angle, ISO/exposure, white balance;
- anamorphic squeeze, lens distortion, breathing, vignetting;
- depth of field and motion blur;
- safe frame, crop, output aspect, and resolution;
- optional rolling shutter.

Camera paths are checked for collision, occlusion, acceleration, angular velocity, focus continuity, and safe subject framing.

## 7. Lighting and colour

The lighting system is scene-referred and physically based:

- OpenColorIO-managed linear workflow with ACES-compatible transforms;
- physical light units;
- sun/sky, HDRI, IES profiles, emissive practicals, and area/spot lights;
- volumetrics, fog, bounce, reflection/refraction, and shadow controls;
- light linking and character key/fill/rim rigs;
- environment, weather, and time-of-day state stored in the world bible;
- render-pass and display transforms recorded in the take manifest.

Users can lock lighting, lens, colour, wardrobe, props, or camera properties. Locked values override AI suggestions.

## 8. Cinematography planner

The director proposes blocking and coverage while enforcing:

- 180-degree axis and screen direction;
- eyelines and gaze targets;
- character/wardrobe/prop continuity;
- lens, depth-of-field, exposure, and light continuity;
- shot-size progression and visual rhythm;
- motivated practical lighting;
- safe camera paths and subject visibility;
- an action beginning, development, and end within the clip duration.

The planner outputs an editable shot specification rather than a final prompt string.

## 9. Rendering modes

### Neural quick render

Text, image, and reference analysis → Wan/LTX worker → optional performance/audio → finish/QC.

Use for ideation, atmospheric shots, or work that does not require exact geometry.

### Authoritative 3D render

USD world → approved physics bake → Blender Cycles/Eevee or engine connector → native 4K frames → audio/finish/QC.

Use for collision, repeatability, exact camera motion, product geometry, and persistent worlds.

### 3D-guided neural render

USD/physics → depth, normals, albedo, segmentation, optical flow, motion vectors, object IDs, shadow, and rough beauty passes → supported neural control/refinement adapter → temporal consistency → finish/QC.

This is the preferred cinematic hybrid: world motion and camera are authoritative while the neural stage adds realism or style.

### Character performance render

Character image/3D render + voice + motion/pose → Wan S2V when eligible, otherwise face/performance animation and lip sync → composite → finish/QC.

## 10. Renderer outputs

Every 3D shot may produce:

- beauty/combined;
- diffuse, specular, emission, transmission, and volume;
- depth and normals;
- segmentation/object/character IDs;
- motion vectors and optical flow;
- shadows, ambient occlusion, cryptomatte;
- world position and UV where required;
- collision/contact debug overlay;
- camera and light metadata.

Control passes are versioned artifacts and can be reused by neural models, compositing, QC, and rerenders without rerunning simulation.