"""Aggregate health check: environment, engine imports, attention backend,
model presence, project folders. Exit code 0 = engine can run something
(at minimum: fabric dry-run + mock engine)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .cuda_checker import check_cuda
from .dependency_checker import check_all
from .ffmpeg_checker import check_ffmpeg
from .torch_checker import check_torch


def run_health_check(ckpt_dir: str = None,
                     project_root: str = ".") -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dependencies": check_all(),
        "torch": check_torch(),
        "cuda": check_cuda(),
        "ffmpeg": check_ffmpeg(),
    }
    try:
        from ..fabric_hooks import attention_backend_report
        import wan  # noqa: F401 — proves base engine imports
        from wan.modules.model import WanModel  # noqa: F401
        report["wan_engine"] = {"imports": True,
                                "attention": attention_backend_report()}
    except Exception as e:
        report["wan_engine"] = {"imports": False, "error": str(e)[:300]}

    if ckpt_dir:
        from .model_downloader import verify_local_model
        report["model"] = verify_local_model("ti2v-5B", ckpt_dir)

    root = Path(project_root)
    report["folders"] = {
        name: (root / name).is_dir()
        for name in ("models", "projects", "configs")
    }

    deps_ok = all(m["found"] for m in report["dependencies"]["modules"].values())
    report["ok"] = bool(report["wan_engine"].get("imports")) and deps_ok
    report["generation_ready"] = bool(
        report["cuda"]["available"] and ckpt_dir and
        report.get("model", {}).get("present"))
    report["mock_ready"] = report["ok"]
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Wan Cognitive Fabric health check")
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--project_root", default=".")
    args = parser.parse_args()
    report = run_health_check(args.ckpt_dir, args.project_root)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
