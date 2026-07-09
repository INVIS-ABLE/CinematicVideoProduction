"""Stitcher (spec §17): chunks → shots → scenes → film via local ffmpeg.
Phase 1 implements lossless-ish concat and fade/dissolve transitions; the
full transition vocabulary maps through TRANSITION_FILTERS as it lands."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

TRANSITION_FILTERS = {
    "cut": None,
    "hard_cut": None,
    "fade_in": "fade=t=in:st=0:d=0.5",
    "fade_out": "fade=t=out:st={fade_start}:d=0.5",
    "dissolve": "xfade=transition=fade:duration=0.5:offset={offset}",
    "dip_to_black": "xfade=transition=fadeblack:duration=0.6:offset={offset}",
    "wipe": "xfade=transition=wipeleft:duration=0.4:offset={offset}",
}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def concat_clips(paths: List[str], output: str) -> str:
    """Concat same-codec clips via the concat demuxer (no re-encode)."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in paths:
            fh.write(f"file '{Path(p).resolve()}'\n")
        list_file = fh.name
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c", "copy", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(list_file).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-800:]}")
    return str(out)


def crossfade_pair(a: str, b: str, output: str, *, fps: int,
                   a_frames: int, duration: float = 0.5,
                   transition: str = "fade") -> str:
    """Re-encoding xfade between two clips (dissolves, dips)."""
    offset = max(0.0, a_frames / fps - duration)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex",
           f"xfade=transition={transition}:duration={duration}:offset={offset:.3f}",
           "-pix_fmt", "yuv420p", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg xfade failed: {result.stderr[-800:]}")
    return str(out)
