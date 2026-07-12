"""Safe local project, asset and job persistence."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import (
    IMAGE_EXTENSIONS,
    MAX_IMAGES,
    MAX_UPLOAD_BYTES,
    MAX_VIDEOS,
    MAX_VOICE_SAMPLES,
    VIDEO_EXTENSIONS,
    VOICE_EXTENSIONS,
)
from .models import AssetKind, AssetRecord, JobRecord, ProjectRecord, utc_now

_PROJECT_RE = re.compile(r"^p_[a-f0-9]{12}$")
_ASSET_RE = re.compile(r"^a_[a-f0-9]{16}$")
_JOB_RE = re.compile(r"^j_[a-f0-9]{16}$")

_LIMITS = {
    AssetKind.image: MAX_IMAGES,
    AssetKind.video: MAX_VIDEOS,
    AssetKind.voice: MAX_VOICE_SAMPLES,
}
_EXTENSIONS = {
    AssetKind.image: IMAGE_EXTENSIONS,
    AssetKind.video: VIDEO_EXTENSIONS,
    AssetKind.voice: VOICE_EXTENSIONS,
}


class StorageError(RuntimeError):
    pass


class StudioStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _new_id(prefix: str, length: int) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:length]}"

    def _project_dir(self, project_id: str) -> Path:
        if not _PROJECT_RE.fullmatch(project_id):
            raise StorageError("invalid project id")
        return self._inside_root(self.root / project_id)

    def _job_dir(self, project_id: str, job_id: str) -> Path:
        if not _JOB_RE.fullmatch(job_id):
            raise StorageError("invalid job id")
        return self._inside_root(self._project_dir(project_id) / "jobs" / job_id)

    def _inside_root(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise StorageError("path escapes studio root") from exc
        return resolved

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise StorageError(f"missing file: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def create_project(self, title: str = "Untitled project") -> ProjectRecord:
        with self._lock:
            project_id = self._new_id("p", 12)
            project = ProjectRecord(project_id=project_id, title=(title.strip() or "Untitled project"))
            project_dir = self._project_dir(project_id)
            for sub in ("assets/image", "assets/video", "assets/voice", "jobs"):
                (project_dir / sub).mkdir(parents=True, exist_ok=True)
            self._write_json(project_dir / "project.json", project.model_dump(mode="json"))
            return project

    def get_project(self, project_id: str) -> ProjectRecord:
        with self._lock:
            payload = self._read_json(self._project_dir(project_id) / "project.json")
            return ProjectRecord.model_validate(payload)

    def _save_project(self, project: ProjectRecord) -> None:
        project.updated_at = utc_now()
        self._write_json(
            self._project_dir(project.project_id) / "project.json",
            project.model_dump(mode="json"),
        )

    async def save_upload(self, project_id: str, kind: AssetKind, upload: Any) -> AssetRecord:
        """Stream one FastAPI UploadFile to disk while enforcing hard limits."""
        with self._lock:
            project = self.get_project(project_id)
            current = sum(1 for asset in project.assets if asset.kind == kind)
            limit = _LIMITS[kind]
            if current >= limit:
                raise StorageError(f"{kind.value} limit reached ({limit})")

        original_name = Path(upload.filename or f"upload{next(iter(_EXTENSIONS[kind]))}").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in _EXTENSIONS[kind]:
            allowed = ", ".join(sorted(_EXTENSIONS[kind]))
            raise StorageError(f"unsupported {kind.value} extension {suffix or '<none>'}; allowed: {allowed}")

        asset_id = self._new_id("a", 16)
        relative_path = Path("assets") / kind.value / f"{asset_id}{suffix}"
        destination = self._inside_root(self._project_dir(project_id) / relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        max_bytes = MAX_UPLOAD_BYTES[kind.value]
        try:
            with destination.open("wb") as fh:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise StorageError(
                            f"{kind.value} exceeds local upload limit of {max_bytes // (1024 * 1024)} MiB"
                        )
                    fh.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            try:
                await upload.close()
            except Exception:
                pass

        metadata = inspect_media(destination, kind)
        record = AssetRecord(
            asset_id=asset_id,
            project_id=project_id,
            kind=kind,
            original_name=original_name,
            relative_path=relative_path.as_posix(),
            media_url=f"/api/projects/{project_id}/assets/{asset_id}/media",
            content_type=getattr(upload, "content_type", None),
            size_bytes=size,
            metadata=metadata,
        )
        with self._lock:
            project = self.get_project(project_id)
            current = sum(1 for asset in project.assets if asset.kind == kind)
            if current >= _LIMITS[kind]:
                destination.unlink(missing_ok=True)
                raise StorageError(f"{kind.value} limit reached ({_LIMITS[kind]})")
            project.assets.append(record)
            self._save_project(project)
        return record

    def get_asset(self, project_id: str, asset_id: str) -> AssetRecord:
        if not _ASSET_RE.fullmatch(asset_id):
            raise StorageError("invalid asset id")
        project = self.get_project(project_id)
        for asset in project.assets:
            if asset.asset_id == asset_id:
                return asset
        raise StorageError("asset not found")

    def asset_path(self, project_id: str, asset_id: str) -> Path:
        asset = self.get_asset(project_id, asset_id)
        path = self._inside_root(self._project_dir(project_id) / asset.relative_path)
        if not path.is_file():
            raise StorageError("asset file not found")
        return path

    def delete_asset(self, project_id: str, asset_id: str) -> None:
        with self._lock:
            project = self.get_project(project_id)
            match = next((a for a in project.assets if a.asset_id == asset_id), None)
            if match is None:
                raise StorageError("asset not found")
            path = self._inside_root(self._project_dir(project_id) / match.relative_path)
            path.unlink(missing_ok=True)
            project.assets = [asset for asset in project.assets if asset.asset_id != asset_id]
            self._save_project(project)

    def resolve_assets(self, project_id: str, asset_ids: Iterable[str]) -> List[AssetRecord]:
        project = self.get_project(project_id)
        by_id = {asset.asset_id: asset for asset in project.assets}
        result: List[AssetRecord] = []
        for asset_id in asset_ids:
            if asset_id not in by_id:
                raise StorageError(f"asset not found in project: {asset_id}")
            result.append(by_id[asset_id])
        return result

    def create_job(self, job: JobRecord) -> JobRecord:
        with self._lock:
            job_dir = self._job_dir(job.project_id, job.job_id)
            job_dir.mkdir(parents=True, exist_ok=False)
            (job_dir / "output").mkdir(exist_ok=True)
            self.save_job(job)
            return job

    def save_job(self, job: JobRecord) -> None:
        job.updated_at = utc_now()
        self._write_json(
            self._job_dir(job.project_id, job.job_id) / "job.json",
            job.model_dump(mode="json"),
        )

    def get_job(self, project_id: str, job_id: str) -> JobRecord:
        payload = self._read_json(self._job_dir(project_id, job_id) / "job.json")
        return JobRecord.model_validate(payload)

    def find_job(self, job_id: str) -> JobRecord:
        if not _JOB_RE.fullmatch(job_id):
            raise StorageError("invalid job id")
        for project_dir in self.root.glob("p_*"):
            job_file = project_dir / "jobs" / job_id / "job.json"
            if job_file.is_file():
                return JobRecord.model_validate(self._read_json(job_file))
        raise StorageError("job not found")

    def list_jobs(self, project_id: str) -> List[JobRecord]:
        jobs_dir = self._project_dir(project_id) / "jobs"
        records = []
        for job_file in jobs_dir.glob("j_*/job.json"):
            try:
                records.append(JobRecord.model_validate(self._read_json(job_file)))
            except Exception:
                continue
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def job_dir(self, project_id: str, job_id: str) -> Path:
        return self._job_dir(project_id, job_id)

    def output_path(self, job: JobRecord) -> Path:
        if not job.output_path:
            raise StorageError("job has no output")
        path = Path(job.output_path)
        if not path.is_absolute():
            path = self._job_dir(job.project_id, job.job_id) / path
        path = self._inside_root(path)
        if not path.is_file():
            raise StorageError("output file not found")
        return path


def inspect_media(path: Path, kind: AssetKind) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if kind == AssetKind.image:
        try:
            from PIL import Image

            with Image.open(path) as image:
                metadata.update(
                    width=image.width,
                    height=image.height,
                    format=image.format,
                    mode=image.mode,
                )
        except Exception as exc:
            metadata["inspection_warning"] = str(exc)
        return metadata

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        metadata["inspection_warning"] = "ffprobe not found"
        return metadata
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        payload = json.loads(result.stdout or "{}")
        metadata["duration"] = float(payload.get("format", {}).get("duration", 0) or 0)
        metadata["streams"] = payload.get("streams", [])
    except Exception as exc:
        metadata["inspection_warning"] = str(exc)
    return metadata
