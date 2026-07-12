# Cinematic Studio Local

Cinematic Studio Local is a private browser-based creative workspace for the Wan 2.2 Cognitive Fabric engine. It uses a modern multimodal creation flow inspired by leading director-style video tools while remaining a distinct local application.

## Product limits

The limits are enforced in both the browser and the API:

- **15 image references**
- **3 video references**
- **3 voice/audio samples**
- **4–15 second clips**
- 480p or 720p native generation
- 16:9, 9:16, 1:1, 4:5 and 2.35:1 framing

Uploads and generated projects stay under `projects/studio/` by default. The app does not call a hosted generation API.

## Install and run

```bash
pip install -r requirements_studio.txt
python run_studio.py
```

Open `http://127.0.0.1:7860`.

The complete interface and job flow work without weights by selecting the deterministic mock engine. This is useful for installation checks, product demos and pipeline testing.

## Real local Wan TI2V generation

Download the Wan 2.2 TI2V-5B checkpoint separately, then point the studio to the local directory:

```bash
# Linux/macOS
export CINESTUDIO_TI2V_CKPT=/absolute/path/to/Wan2.2-TI2V-5B
python run_studio.py

# PowerShell
$env:CINESTUDIO_TI2V_CKPT="D:\models\Wan2.2-TI2V-5B"
python run_studio.py
```

A CUDA GPU and the normal Wan dependencies are required. The studio automatically falls back to mock mode when CUDA or the checkpoint is unavailable.

## Voice-driven Performance mode

Performance mode can use Wan S2V when all of the following are present:

1. an opening image;
2. an active voice/audio sample;
3. a local Wan2.2-S2V-14B checkpoint;
4. the S2V optional dependencies and CUDA.

Configure it with:

```bash
export CINESTUDIO_S2V_CKPT=/absolute/path/to/Wan2.2-S2V-14B
python run_studio.py
```

The selected video reference can be passed to S2V as pose/motion guidance. When S2V is not available, the studio uses TI2V or mock generation and can mux the active audio sample into the finished MP4 with FFmpeg.

## How references are used

- The selected **opening image** is directly pixel-conditioned through Wan TI2V's image-to-video path.
- In **Performance mode**, the selected voice sample and optional motion video are used directly by Wan S2V when that checkpoint is installed.
- Every uploaded reference receives a role and an optional note. Notes are compiled into the generation prompt and all references are recorded in the local project manifest.
- Additional references also enter the Cognitive Fabric reference-token registry. The current open Wan TI2V model does not provide Seedance-style native joint conditioning over all images, videos and audio clips, so the interface labels this as a local approximation rather than claiming equivalent model capability.

For the most reliable identity control, make the strongest character or subject image the opening frame and describe exactly what must remain unchanged in its reference note.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CINESTUDIO_PROJECTS_DIR` | `projects/studio` | Project and upload storage |
| `CINESTUDIO_TI2V_CKPT` | unset | Wan2.2-TI2V-5B checkpoint |
| `CINESTUDIO_S2V_CKPT` | unset | Wan2.2-S2V-14B checkpoint |
| `CINESTUDIO_HOST` | `127.0.0.1` | Local bind address |
| `CINESTUDIO_PORT` | `7860` | Local port |
| `CINESTUDIO_FORCE_MOCK` | `false` | Force the deterministic mock engine |
| `CINESTUDIO_LOCAL_ONLY` | `true` | Report and enforce local-first operation |
| `CINESTUDIO_MAX_WORKERS` | `1` | Concurrent generation workers |

Keep `CINESTUDIO_MAX_WORKERS=1` for a single GPU. Multiple simultaneous diffusion jobs usually increase failures and reduce throughput.

## Safety and consent

Use only images, video, voices and characters you own or have permission to use. Obtain explicit consent before generating with another person's face or voice. Generated media should be labelled appropriately when realism could mislead viewers.
