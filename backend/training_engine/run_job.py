"""Child-process entry point: python -m backend.training_engine.run_job <job_id>"""
from __future__ import annotations

import sys
import traceback

from backend import schemas
from backend.infrastructure import storage
from backend.training_engine import trainer


def main(job_id: str) -> int:
    job = storage.load("jobs", job_id, schemas.TrainingJob)
    if job is None:
        print(f"job {job_id} not found", file=sys.stderr)
        return 2

    def progress(p: float, msg: str) -> None:
        job.progress, job.message = round(p, 3), msg
        storage.save("jobs", job)

    try:
        req = schemas.TrainingRequest(**job.request)
        ds = storage.load("datasets", req.dataset_id, schemas.DatasetManifest)
        if ds is None:
            raise ValueError(f"dataset {req.dataset_id} not found")
        art = trainer.train(req, ds, progress)
        storage.save("models", art)
        job.result_id = art.id
        job.status = schemas.JobStatus.succeeded
        job.progress, job.message = 1.0, f"done in {art.train_seconds}s, metrics {art.metrics}"
    except Exception as e:  # report every failure on the job, never silently
        job.status = schemas.JobStatus.failed
        job.error = f"{type(e).__name__}: {e}"
        job.message = traceback.format_exc()[-2000:]
    job.finished_at = schemas.now()
    storage.save("jobs", job)
    return 0 if job.status == schemas.JobStatus.succeeded else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
