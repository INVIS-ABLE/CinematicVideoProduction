"""FFmpeg checker."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict


def check_ffmpeg() -> Dict[str, Any]:
    path = shutil.which("ffmpeg")
    if path is None:
        return {"found": False,
                "note": "stitching/export disabled until ffmpeg is installed"}
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                             text=True, timeout=10)
        version = out.stdout.splitlines()[0] if out.stdout else "unknown"
    except Exception as e:
        version = f"error: {e}"
    return {"found": True, "path": path, "version": version}
