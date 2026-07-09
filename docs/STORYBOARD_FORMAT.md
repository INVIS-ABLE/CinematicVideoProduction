# Storyboard Format

Schema: `configs/storyboard_schema.json`. A storyboard is the complete film
plan: project → characters/objects/worlds → acts → scenes → shots. The
Director Brain generates one automatically from a prompt (`cognitive-short`);
`cognitive-film` consumes a hand-written or tool-written one.

## Minimal working example

```json
{
  "project": {"title": "Neon Rain", "style": "ultra realistic cinematic",
               "frame_rate": 24, "aspect_ratio": "16:9"},
  "characters": [{
    "character_id": "char_001", "name": "the hero",
    "face_description": "weathered face, scarred jaw",
    "wardrobe_description": "soaked black coat"
  }],
  "objects": [{"object_id": "obj_001", "name": "damaged helmet",
                "damage_state": "dented"}],
  "worlds": [{"world_id": "world_001",
               "description": "ruined futuristic city street",
               "location_type": "urban cityscape", "time_of_day": "night",
               "weather": "rain", "light_direction": "motivated by signage"}],
  "acts": [{"act_id": "act_001", "scenes": ["scene_001"]}],
  "scenes": [{"scene_id": "scene_001", "act_id": "act_001",
               "world_id": "world_001",
               "purpose": "introduce the hero and the city",
               "shots": ["shot_001", "shot_002"]}],
  "shots": [
    {"shot_id": "shot_001", "scene_id": "scene_001", "duration_seconds": 5,
     "prompt": "wide aerial of a ruined futuristic city at night in rain",
     "shot_type": "establishing wide", "world_id": "world_001",
     "camera_plan": {"movement": "crane_down", "lens": "24mm"},
     "transition_in": "fade_in"},
    {"shot_id": "shot_002", "scene_id": "scene_001", "duration_seconds": 5,
     "prompt": "medium rear tracking shot of the hero walking through rain",
     "shot_type": "medium tracking", "world_id": "world_001",
     "character_ids": ["char_001"], "object_ids": ["obj_001"],
     "camera_plan": {"movement": "tracking", "lens": "35mm"},
     "continuity_from_previous": true, "transition_out": "fade_out"}
  ]
}
```

## Semantics

- `continuity_from_previous: true` → the chunk opens on the previous chunk's
  terminal frame (Wan TI2V latent clamp) and inherits world/lighting clauses.
- `seed_strategy`: `fixed` (project seed), `per_shot` (stable per shot id),
  `per_chunk` (varies within a shot).
- Durations are snapped to Wan's 4n+1 frame invariant per chunk; long shots
  split at `cognitive_fabric.chunk_seconds` with `overlap_frames` recorded.
- Camera movements map to precise prompt language in
  `wan/cognitive_fabric/conditioning/camera_guidance.py` (static, dolly_in,
  dolly_out, orbit, crane_up, crane_down, tracking, handheld, low_angle,
  reveal, parallax_pan, over_shoulder, push_in).
