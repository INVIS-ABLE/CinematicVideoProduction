"""Fabric logging: console + per-project file logs under <project>/logs/."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "[%(asctime)s] fabric.%(name)s %(levelname)s: %(message)s"


def get_fabric_logger(name: str = "core",
                      project_dir: Optional[str] = None,
                      level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"cognitive_fabric.{name}")
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and
               getattr(h, "_fabric_console", False) for h in logger.handlers):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT))
        console._fabric_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)

    if project_dir:
        log_dir = Path(project_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "fabric.log"
        if not any(
                isinstance(h, logging.FileHandler) and
                Path(getattr(h, "baseFilename", "")) == log_file
                for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)
    return logger
