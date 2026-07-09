"""Windows install helper invoked by installers/windows_one_click_setup.ps1:
venv creation and requirement installation live in the PowerShell script;
this module performs the Python-side setup and verification steps."""
from __future__ import annotations

from .first_run_setup import main

if __name__ == "__main__":
    raise SystemExit(main())
