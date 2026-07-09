"""Concurrent model runner (spec §20): schedules jobs across models with a
VRAM-aware budget. Phase 1 runs a real thread-pool job queue with
pause/resume and budget accounting; multi-model GPU residency policies
deepen in Phase 8."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(order=True)
class Job:
    priority: int
    created_at: float = field(compare=False, default_factory=time.time)
    job_id: str = field(compare=False, default="")
    kind: str = field(compare=False, default="generate")
    vram_gb: float = field(compare=False, default=0.0)
    fn: Optional[Callable[[], Any]] = field(compare=False, default=None)


class ConcurrentModelRunner:
    def __init__(self, vram_budget_gb: float = 8.0, workers: int = 2):
        self.vram_budget_gb = vram_budget_gb
        self._queue: "queue.PriorityQueue[Job]" = queue.PriorityQueue()
        self._results: Dict[str, Any] = {}
        self._vram_in_use = 0.0
        self._lock = threading.Lock()
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._workers = [threading.Thread(target=self._worker, daemon=True)
                         for _ in range(workers)]
        for w in self._workers:
            w.start()

    def submit(self, job_id: str, fn: Callable[[], Any], *, kind: str = "generate",
               priority: int = 5, vram_gb: float = 0.0) -> None:
        self._queue.put(Job(priority=priority, job_id=job_id, kind=kind,
                            vram_gb=vram_gb, fn=fn))

    def _worker(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                if self._vram_in_use + job.vram_gb > self.vram_budget_gb:
                    self._queue.put(job)  # requeue until budget frees up
                    time.sleep(0.05)
                    continue
                self._vram_in_use += job.vram_gb
            try:
                self._results[job.job_id] = {"ok": True, "value": job.fn()}
            except Exception as e:
                self._results[job.job_id] = {"ok": False, "error": str(e)}
            finally:
                with self._lock:
                    self._vram_in_use -= job.vram_gb
                self._queue.task_done()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def wait(self, timeout: Optional[float] = None) -> None:
        deadline = time.time() + timeout if timeout else None
        while not self._queue.empty():
            if deadline and time.time() > deadline:
                raise TimeoutError("jobs still pending")
            time.sleep(0.02)

    def result(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._results.get(job_id)

    def shutdown(self) -> None:
        self._stop.set()
