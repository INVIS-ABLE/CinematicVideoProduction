"""Export engine: ffmpeg encode presets + audio mux for final delivery."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

PRESETS: Dict[str, List[str]] = {
    "draft": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"],
    "review": ["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
    "master": ["-c:v", "libx264", "-preset", "slow", "-crf", "14"],
    "intermediate": ["-c:v", "libx264", "-preset", "slow", "-crf", "8"],
}


def export(input_path: str, output_path: str, preset: str = "review",
           fps: Optional[int] = None, audio_path: Optional[str] = None) -> str:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found — install it or place it on PATH")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if audio_path:
        cmd += ["-i", audio_path, "-map", "0:v", "-map", "1:a", "-shortest",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += PRESETS.get(preset, PRESETS["review"])
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-pix_fmt", "yuv420p", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"export failed: {result.stderr[-800:]}")
    return str(out)
