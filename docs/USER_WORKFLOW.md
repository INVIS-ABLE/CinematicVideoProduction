# User Workflow

## 1. Install (Windows)

```powershell
git clone <this repo>
cd CinematicVideoProduction
.\installers\windows_one_click_setup.ps1            # env + health check
.\installers\windows_one_click_setup.ps1 -Download  # + fetch TI2V-5B weights
```

Linux/macOS: create a venv, `pip install -r requirements.txt` (skip
`flash_attn`/`dashscope` if they fail — both optional now), then
`pip install -r requirements_cognitive.txt`.

## 2. Check the engine

```bash
python -m wan.cognitive_fabric.install.health_check --ckpt_dir ./models/Wan2.2-TI2V-5B
```

`mock_ready: true` → the whole pipeline runs anywhere.
`generation_ready: true` → real Wan generation is available.

## 3. Create

One idea → short cinematic sequence (Director Brain plans the shots):

```bash
python generate.py --task cognitive-short \
  --prompt "Create a cinematic ultra-realistic scene of a futuristic city at night with a damaged hero walking through rain, neon reflections, slow camera push-in, blade-runner lighting, maintain character continuity" \
  --negative_prompt "cartoon, warped face, identity drift" \
  --duration_seconds 30 \
  --ckpt_dir ./models/Wan2.2-TI2V-5B
```

Full storyboard film (see docs/STORYBOARD_FORMAT.md):

```bash
python generate.py --task cognitive-film --storyboard ./projects/my_film/storyboard.json \
  --ckpt_dir ./models/Wan2.2-TI2V-5B --final_resolution 4k
```

Anime episode with series memory:

```bash
python generate.py --task cognitive-anime-episode \
  --series_bible ./projects/anime/series_bible.json \
  --episode ./projects/anime/episode_01.json \
  --ckpt_dir ./models/Wan2.2-TI2V-5B --anime_mode true
```

Useful flags: `--dry_run true` (plan only), `--force_mock_engine true`
(pipeline test), `--fabric_config configs/cognitive_fabric.low_vram.yaml`,
`--project_dir projects/my_project` (resume by re-running the same command).

## 4. Results

Everything lands in `projects/<id>/`: `exports/draft_assembly.mp4`,
per-chunk clips + quality reports (`memory.db`), `timeline.json`,
`run_report.json`, and checkpoints for resume.

## 5. Classic Wan (unchanged)

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./models/Wan2.2-TI2V-5B --prompt "..."
```
