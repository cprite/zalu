"""Orchestration Engine: queue jobs, run one GPU job at a time in a child process, track status."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading

from backend import schemas
from backend.infrastructure import storage

_q: "queue.Queue[str]" = queue.Queue()
_started = False


def submit_training(req: schemas.TrainingRequest) -> schemas.TrainingJob:
    if storage.load("datasets", req.dataset_id, schemas.DatasetManifest) is None:
        raise ValueError(f"dataset {req.dataset_id} not found")
    job = schemas.TrainingJob(kind="training", request=req.model_dump())
    storage.save("jobs", job)
    _q.put(job.id)
    return job


def _worker() -> None:
    while True:
        job_id = _q.get()
        job = storage.load("jobs", job_id, schemas.TrainingJob)
        if job is None:
            continue
        job.status, job.started_at, job.message = schemas.JobStatus.running, schemas.now(), "starting"
        storage.save("jobs", job)
        log = storage.path("runs", f"{job_id}.log")
        with log.open("w") as fh:
            rc = subprocess.call(
                [sys.executable, "-m", "backend.training_engine.run_job", job_id],
                stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy(),
            )
        job = storage.load("jobs", job_id, schemas.TrainingJob)
        if job and job.status == schemas.JobStatus.running:  # child died without writing a result
            job.status, job.error = schemas.JobStatus.failed, f"worker exited with code {rc}, see {log}"
            job.finished_at = schemas.now()
            storage.save("jobs", job)


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    # jobs left 'running' by a previous crash are marked failed, queued ones are resumed
    for job in storage.list_all("jobs", schemas.TrainingJob):
        if job.status == schemas.JobStatus.running:
            job.status, job.error = schemas.JobStatus.failed, "server restarted during the job"
            storage.save("jobs", job)
        elif job.status == schemas.JobStatus.queued:
            _q.put(job.id)
    threading.Thread(target=_worker, daemon=True, name="gpu-worker").start()
