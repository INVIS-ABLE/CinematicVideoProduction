"""Build and optionally wrap the Windows desktop bundle in an Inno Setup installer."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    run(sys.executable, "-m", "pytest", "-q", "tests/local_studio")
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "packaging/CinematicStudio.spec")
    bundle = ROOT / "dist" / "CinematicStudio"
    tools = bundle / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable:
            shutil.copy2(executable, tools / Path(executable).name)
    iscc = shutil.which("ISCC.exe") or shutil.which("iscc")
    if iscc:
        run(iscc, "packaging/windows/CinematicStudio.iss")
    else:
        print("Inno Setup not found; onedir bundle is ready at", bundle)


if __name__ == "__main__":
    main()
