"""First-run setup: creates folders, writes default config + model registry,
runs the health check, optionally downloads the selected model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..fabric_config import DEFAULT_CONFIG
from .health_check import run_health_check
from .model_downloader import download_model


def first_run_setup(root: str = ".", *, download: bool = False,
                    task: str = "ti2v-5B") -> Dict[str, Any]:
    base = Path(root)
    for folder in ("models", "projects", "configs", "logs"):
        (base / folder).mkdir(parents=True, exist_ok=True)

    config_path = base / "configs" / "cognitive_fabric.local.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2),
                               encoding="utf-8")

    registry_path = base / "models" / "model_registry.json"
    if not registry_path.exists():
        registry_path.write_text(json.dumps({"models": []}, indent=2),
                                 encoding="utf-8")

    result: Dict[str, Any] = {
        "config": str(config_path),
        "model_registry": str(registry_path),
    }
    if download:
        result["download"] = download_model(
            task, str(base / "models" / f"Wan2.2-{task}"), allow_network=True)
    result["health"] = run_health_check(project_root=root)
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive Fabric first-run setup")
    parser.add_argument("--root", default=".")
    parser.add_argument("--download", action="store_true",
                        help="download the selected Wan model (network use "
                             "is opt-in; generation stays local)")
    parser.add_argument("--task", default="ti2v-5B")
    args = parser.parse_args()
    result = first_run_setup(args.root, download=args.download, task=args.task)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["health"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
