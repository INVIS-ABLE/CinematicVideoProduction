# Wan 2.2 Cognitive Fabric — Windows one-click setup.
# Usage (PowerShell):  .\installers\windows_one_click_setup.ps1 [-Download] [-Task ti2v-5B]
# Checks Python/Git/FFmpeg, creates a venv, installs requirements, runs the
# health check, and (optionally, with -Download) fetches the selected open
# Wan model from Hugging Face. Generation itself is always local.
param(
    [switch]$Download,
    [string]$Task = "ti2v-5B",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
Write-Host "== Wan 2.2 Cognitive Fabric setup ==" -ForegroundColor Cyan
Write-Host "repo: $Root"

# 1. Python
try { $pyver = & $PythonExe --version } catch {
    Write-Error "Python not found. Install Python 3.10+ from python.org and re-run."
}
Write-Host "python: $pyver"

# 2. Git / FFmpeg (informational — ffmpeg needed for stitching/export)
foreach ($tool in @("git", "ffmpeg")) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Write-Host ("{0}: found" -f $tool)
    } else {
        Write-Warning ("{0}: NOT found — install it for full functionality" -f $tool)
    }
}

# 3. venv
if (-not (Test-Path ".venv")) {
    & $PythonExe -m venv .venv
    Write-Host "created .venv"
}
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"

# 4. PyTorch (CUDA build if an NVIDIA GPU is present, else CPU build)
& $venvPy -m pip install --upgrade pip
$hasNvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($hasNvidia) {
    Write-Host "NVIDIA GPU detected - installing CUDA PyTorch (cu124)"
    & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Warning "no NVIDIA GPU detected - installing CPU PyTorch (mock/dry-run only)"
    & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# 5. Requirements (flash_attn is optional on Windows; skip failures gracefully)
Get-Content requirements.txt | Where-Object {$_ -notmatch "^(flash_attn|torch|torchvision|torchaudio|dashscope)"} |
    Set-Content "$env:TEMP\wan_reqs.txt"
& $venvPy -m pip install -r "$env:TEMP\wan_reqs.txt"
& $venvPy -m pip install -r requirements_cognitive.txt
try {
    & $venvPy -m pip install flash_attn --no-build-isolation
} catch {
    Write-Warning "flash_attn unavailable - engine uses the SDPA fallback (slower, still correct)"
}

# 6. First-run setup + health check (+ optional model download)
$setupArgs = @("-m", "wan.cognitive_fabric.install.first_run_setup", "--root", ".")
if ($Download) { $setupArgs += @("--download", "--task", $Task) }
& $venvPy @setupArgs

# 7. Smoke test: full fabric dry run (no GPU, no weights required)
& $venvPy generate.py --task cognitive-short --prompt "setup smoke test: a lantern in the rain at night" --duration_seconds 8 --dry_run true --project_dir projects\_setup_smoke
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSetup complete. Try:" -ForegroundColor Green
    Write-Host "  .venv\Scripts\python generate.py --task cognitive-short --prompt `"...`" --duration_seconds 20"
    Write-Host "Add --ckpt_dir models\Wan2.2-TI2V-5B for real generation (CUDA GPU required)."
} else {
    Write-Error "smoke test failed - see output above"
}
