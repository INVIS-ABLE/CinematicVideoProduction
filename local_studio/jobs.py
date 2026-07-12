"""Single-GPU friendly persistent job queue."""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

from .generation import StudioGenerator
from .models import GenerationRequest, JobRecord, JobStatus
from .storage import StudioStorage


class JobManager:
    def __init__(
        self,
        storage: StudioStorage,
        generator: StudioGenerator,
        max_workers: int = 1,
    ):
        self.storage = storage
        self.generator = generator
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="cinestudio",
        )
        self._lock = threading.RLock()
        self._futures: Dict[str, Future] = {}

    def submit(self, request: GenerationRequest) -> JobRecord:
        # Resolve now so malformed project/asset IDs fail before a job is queued.
        self.storage.get_project(request.project_id)
        self.storage.resolve_assets(
            request.project_id,
            request.image_ids + request.video_ids + request.voice_ids,
        )
        job_id = f"j_{uuid.uuid4().hex[:16]}"
        job = JobRecord(job_id=job_id, project_id=request.project_id, request=request)
        self.storage.create_job(job)
        future = self.executor.submit(self._run, job)
        with self._lock:
            self._futures[job_id] = future
        return job

    def _run(self, initial: JobRecord) -> None:
        job = initial
        try:
            self._update(job, status=JobStatus.preparing, progress=2, message="Preparing local job")

            def progress(percent: int, message: str) -> None:
                status = JobStatus.generating if percent < 88 else JobStatus.finishing
                self._update(job, status=status, progress=percent, message=message)

            result = self.generator.generate(job.request, job.job_id, progress)
            job.output_path = str(result.output_path.resolve())
            job.output_url = f"/api/jobs/{job.job_id}/output"
            job.report = result.report
            job.error = None
            self._update(
                job,
                status=JobStatus.completed,
                progress=100,
                message="Generation complete",
            )
        except Exception as exc:
            job.error = str(exc)
            job.report = {
                **job.report,
                "traceback": traceback.format_exc(limit=20),
            }
            self._update(
                job,
                status=JobStatus.failed,
                progress=max(job.progress, 1),
                message="Generation failed",
            )
        finally:
            with self._lock:
                self._futures.pop(job.job_id, None)

    def _update(
        self,
        job: JobRecord,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        with self._lock:
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = min(100, max(0, int(progress)))
            if message is not None:
                job.message = message
            self.storage.save_job(job)

    def get(self, job_id: str) -> JobRecord:
        return self.storage.find_job(job_id)

    def list_for_project(self, project_id: str) -> List[JobRecord]:
        return self.storage.list_jobs(project_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
