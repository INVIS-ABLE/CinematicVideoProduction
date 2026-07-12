"""FastAPI entry point for the local-first cinematic studio."""
from __future__ import annotations

import argparse
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    ASPECT_RATIOS,
    MAX_DURATION_SECONDS,
    MAX_IMAGES,
    MAX_VIDEOS,
    MAX_VOICE_SAMPLES,
    MIN_DURATION_SECONDS,
    StudioSettings,
    native_size,
)
from .generation import StudioGenerator
from .jobs import JobManager
from .models import AssetKind, GenerationRequest, JobRecord, ProjectRecord
from .storage import StorageError, StudioStorage

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateProjectRequest(BaseModel):
    title: str = Field(default="Untitled project", max_length=120)


def create_app(settings: Optional[StudioSettings] = None) -> FastAPI:
    settings = settings or StudioSettings.from_env()
    storage = StudioStorage(settings.root_dir)
    generator = StudioGenerator(settings, storage)
    jobs = JobManager(storage, generator, settings.max_workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(
        title="Cinematic Studio Local",
        description="A private local multimodal video studio powered by Wan Cognitive Fabric.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.storage = storage
    app.state.jobs = jobs

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(StorageError)
    async def storage_error_handler(_, exc: StorageError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def status() -> Dict[str, Any]:
        cuda_available = False
        gpu_name = None
        torch_version = None
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        ti2v_ready = cuda_available and settings.checkpoint_exists("ti2v") and not settings.force_mock
        s2v_ready = cuda_available and settings.checkpoint_exists("s2v") and not settings.force_mock
        return {
            "app": "Cinematic Studio Local",
            "version": "0.1.0",
            "local_only": settings.local_only,
            "projects_dir": str(settings.root_dir),
            "limits": {
                "images": MAX_IMAGES,
                "videos": MAX_VIDEOS,
                "voice_samples": MAX_VOICE_SAMPLES,
                "duration_min": MIN_DURATION_SECONDS,
                "duration_max": MAX_DURATION_SECONDS,
            },
            "hardware": {
                "cuda_available": cuda_available,
                "gpu_name": gpu_name,
                "torch_version": torch_version,
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "ffprobe": bool(shutil.which("ffprobe")),
            },
            "models": {
                "ti2v_ready": ti2v_ready,
                "s2v_ready": s2v_ready,
                "mock_fallback": True,
                "selected": "wan-ti2v-5b" if ti2v_ready else "mock",
            },
            "native_sizes": {
                resolution: {
                    ratio: native_size(resolution, ratio)
                    for ratio in ASPECT_RATIOS
                }
                for resolution in ("480p", "720p")
            },
        }

    @app.post("/api/projects", response_model=ProjectRecord)
    async def create_project(payload: CreateProjectRequest) -> ProjectRecord:
        return storage.create_project(payload.title)

    @app.get("/api/projects/{project_id}", response_model=ProjectRecord)
    async def get_project(project_id: str) -> ProjectRecord:
        return storage.get_project(project_id)

    @app.post(
        "/api/projects/{project_id}/assets/{kind}",
        response_model=Any,
    )
    async def upload_asset(
        project_id: str,
        kind: AssetKind,
        file: UploadFile = File(...),
    ):
        return await storage.save_upload(project_id, kind, file)

    @app.delete("/api/projects/{project_id}/assets/{asset_id}")
    async def delete_asset(project_id: str, asset_id: str) -> Dict[str, bool]:
        storage.delete_asset(project_id, asset_id)
        return {"deleted": True}

    @app.get("/api/projects/{project_id}/assets/{asset_id}/media", include_in_schema=False)
    async def asset_media(project_id: str, asset_id: str):
        asset = storage.get_asset(project_id, asset_id)
        path = storage.asset_path(project_id, asset_id)
        return FileResponse(
            path,
            media_type=asset.content_type or "application/octet-stream",
            filename=None,
        )

    @app.post("/api/jobs", response_model=JobRecord)
    async def submit_job(request: GenerationRequest) -> JobRecord:
        return jobs.submit(request)

    @app.get("/api/jobs/{job_id}", response_model=JobRecord)
    async def get_job(job_id: str) -> JobRecord:
        return jobs.get(job_id)

    @app.get("/api/projects/{project_id}/jobs", response_model=List[JobRecord])
    async def list_jobs(project_id: str) -> List[JobRecord]:
        return jobs.list_for_project(project_id)

    @app.get("/api/jobs/{job_id}/output", include_in_schema=False)
    async def job_output(job_id: str):
        job = jobs.get(job_id)
        if not job.output_path:
            raise HTTPException(status_code=404, detail="job output is not ready")
        path = storage.output_path(job)
        media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cinematic Studio Local")
    parser.add_argument("--host", default=None, help="Bind address; defaults to CINESTUDIO_HOST")
    parser.add_argument("--port", type=int, default=None, help="Port; defaults to CINESTUDIO_PORT")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn development reload")
    args = parser.parse_args()

    settings = StudioSettings.from_env()
    host = args.host or settings.host
    port = args.port or settings.port
    import uvicorn

    uvicorn.run(
        "local_studio.app:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
