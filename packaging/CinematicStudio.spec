# PyInstaller onedir specification for the local desktop application.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parent.parent
hidden = collect_submodules("wan.cognitive_fabric", on_error="warn once")
hidden += ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"]
datas = [(str(root / "local_studio" / "static"), "local_studio/static")]
datas += collect_data_files("wan", include_py_files=False)

a = Analysis(
    [str(root / "run_desktop.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=["tkinter", "matplotlib", "notebook", "jupyter", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CinematicStudio", console=False, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="CinematicStudio")
