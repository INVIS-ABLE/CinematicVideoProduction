from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient

from local_studio.app import create_app
from local_studio.config import StudioSettings


def test_status_and_project_creation(tmp_path: Path):
    settings = StudioSettings(
        root_dir=tmp_path / "studio",
        ti2v_checkpoint=None,
        s2v_checkpoint=None,
        force_mock=True,
    )
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["limits"] == {
            "images": 15,
            "videos": 3,
            "voice_samples": 3,
            "duration_min": 4,
            "duration_max": 15,
        }
        project = client.post("/api/projects", json={"title": "Test film"})
        assert project.status_code == 200
        payload = project.json()
        assert payload["title"] == "Test film"
        assert payload["project_id"].startswith("p_")
        assert (settings.root_dir / payload["project_id"] / "project.json").is_file()
