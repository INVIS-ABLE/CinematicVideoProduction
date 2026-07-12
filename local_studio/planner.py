"""Prompt, reference and storyboard planning for the local studio."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import ASPECT_RATIOS, native_size
from .models import AssetDirective, AssetKind, AssetRecord, GenerationRequest

CAMERA_PROMPTS = {
    "director_auto": "director-selected cinematic camera language with motivated cuts",
    "locked": "locked-off tripod camera, deliberate composition, no unwanted camera shake",
    "handheld": "controlled documentary handheld camera with natural micro-movement",
    "dolly_in": "slow precise dolly-in with stable parallax",
    "orbit": "smooth cinematic orbit around the main subject",
    "tracking": "smooth lateral tracking shot that preserves screen direction",
    "crane": "elegant crane movement revealing depth and scale",
}

MOTION_PROMPTS = {
    "subtle": "subtle physically plausible subject motion, restrained background movement",
    "balanced": "clear natural motion with stable anatomy and coherent object interaction",
    "dynamic": "dynamic action with readable silhouettes, grounded momentum and controlled motion blur",
    "slow_motion": "cinematic slow motion with consistent temporal detail",
}

STYLE_PROMPTS = {
    "cinematic": "cinematic production design, nuanced lighting, natural contrast, detailed materials",
    "photoreal": "photorealistic live-action image, physically plausible lighting and texture",
    "anime": "premium cinematic anime, consistent line work, controlled cel shading",
    "commercial": "polished premium advertising photography, precise product detail",
    "documentary": "observational documentary realism, authentic light and texture",
    "noir": "film-noir contrast, motivated practical light, restrained monochrome palette",
}

ROLE_PROMPTS = {
    "opening": "opening composition anchor",
    "character": "character identity and wardrobe reference",
    "subject": "main subject appearance reference",
    "environment": "environment, architecture and world reference",
    "style": "visual style, palette and material reference",
    "composition": "composition and framing reference",
    "motion": "motion rhythm and action reference",
    "camera": "camera path and lens-behaviour reference",
    "performance": "facial performance and body-language reference",
    "voice": "voice, cadence and audio timing reference",
    "soundtrack": "soundtrack and ambience reference",
    "reference": "creative reference",
}


@dataclass
class PlannedInputs:
    prompt: str
    negative_prompt: str
    opening_frame: Optional[Path]
    active_voice: Optional[Path]
    motion_video: Optional[Path]
    reference_manifest: List[Dict[str, str]]


def build_prompt(
    request: GenerationRequest,
    assets: Iterable[AssetRecord],
) -> Tuple[str, List[Dict[str, str]]]:
    """Compile user intent and explicit reference notes without inventing captions."""
    clauses = [request.prompt.rstrip(" .")]
    clauses.append(STYLE_PROMPTS.get(request.style, request.style))
    clauses.append(CAMERA_PROMPTS.get(request.camera, request.camera))
    clauses.append(MOTION_PROMPTS.get(request.motion, request.motion))
    clauses.append(
        "coherent identity, consistent wardrobe and props, stable lighting direction, "
        "physically plausible contact and motion, clean cinematic continuity"
    )

    manifest: List[Dict[str, str]] = []
    for index, asset in enumerate(assets, start=1):
        directive = request.asset_directives.get(asset.asset_id, AssetDirective())
        role_text = ROLE_PROMPTS.get(directive.role, directive.role.replace("_", " "))
        entry = {
            "asset_id": asset.asset_id,
            "kind": asset.kind.value,
            "role": directive.role,
            "note": directive.note.strip(),
        }
        manifest.append(entry)
        if directive.note.strip():
            clauses.append(
                f"Reference {asset.kind.value} {index} is the {role_text}: {directive.note.strip()}"
            )
        else:
            # The model is told the intent, but the app never fabricates visual content from a filename.
            clauses.append(f"Reference {asset.kind.value} {index} is assigned as {role_text}")

    prompt = ". ".join(clause.strip().rstrip(".") for clause in clauses if clause.strip())
    return prompt, manifest


def default_negative_prompt(user_negative: str) -> str:
    base = (
        "identity drift, face morphing, inconsistent clothing, extra limbs, malformed hands, "
        "floating objects, broken contact, background warping, frame flicker, strobing, jitter, "
        "unmotivated camera movement, text, logo, watermark"
    )
    if user_negative.strip():
        return f"{user_negative.strip().rstrip(',')}, {base}"
    return base


def prepare_opening_frame(
    source: Path,
    destination: Path,
    aspect_ratio: str,
    resolution: str,
) -> Path:
    """Crop and resize an opening image to the requested Wan-native aspect."""
    from PIL import Image, ImageOps

    width, height = native_size(resolution, aspect_ratio)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        source_ratio = image.width / max(image.height, 1)
        target_ratio = ASPECT_RATIOS[aspect_ratio]
        if source_ratio > target_ratio:
            crop_width = int(image.height * target_ratio)
            left = max(0, (image.width - crop_width) // 2)
            image = image.crop((left, 0, left + crop_width, image.height))
        else:
            crop_height = int(image.width / target_ratio)
            top = max(0, (image.height - crop_height) // 2)
            image = image.crop((0, top, image.width, top + crop_height))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.save(destination, quality=95)
    return destination


def extract_video_frame(source: Path, destination: Path, position: str = "middle") -> Path:
    """Extract a deterministic reference frame from a local video with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to use a video as the opening reference")

    timestamp = "0"
    if position == "middle" and ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            duration = max(0.0, float(result.stdout.strip() or 0))
            timestamp = f"{duration / 2:.3f}"
        except Exception:
            timestamp = "0"

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        timestamp,
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"could not extract video reference frame: {result.stderr[-600:]}")
    return destination
