"""Crash-isolated generation worker entry point."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import StudioSettings
from .connectors import ConnectorManager
from .generation import StudioGenerator
from .models import GenerationRequest
from .storage import StudioStorage
from .story_memory import StoryMemory


def run(request_file: Path, result_file: Path, progress_file: Path, job_id: str) -> None:
    settings = StudioSettings.from_env()
    storage = StudioStorage(settings.root_dir)
    stories = StoryMemory(settings.story_database)
    connectors = ConnectorManager(settings.connector_config, settings.local_only)
    generator = StudioGenerator(settings, storage, stories, connectors)
    request = GenerationRequest.model_validate_json(request_file.read_text(encoding="utf-8"))

    def progress(percent: int, message: str) -> None:
        temp = progress_file.with_suffix(".tmp")
        temp.write_text(json.dumps({"progress": percent, "message": message}), encoding="utf-8")
        temp.replace(progress_file)

    result = generator.generate(request, job_id, progress)
    temp = result_file.with_suffix(".tmp")
    temp.write_text(json.dumps({"output_path": str(result.output_path.resolve()), "report": result.report}, default=str), encoding="utf-8")
    temp.replace(result_file)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: python -m local_studio.worker REQUEST RESULT PROGRESS JOB_ID")
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])


if __name__ == "__main__":
    main()
