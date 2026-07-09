@echo off
REM Wan 2.2 Cognitive Fabric — portable launcher (run from the repo root).
REM Uses the venv created by windows_one_click_setup.ps1.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo venv missing — run installers\windows_one_click_setup.ps1 first
    exit /b 1
)
".venv\Scripts\python.exe" -m wan.cognitive_fabric.install.health_check
".venv\Scripts\python.exe" generate.py %*
endlocal
