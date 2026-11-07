"""
A small job queue for the retraining loop (Task 9).

The backend does not train or restart containers itself: training needs
Spark, DVC and the Git repository, and restarting a container needs Docker.
Giving the backend container access to Docker would let anyone who breaks
into the API control every container on the laptop.

So the backend only RECORDS work to be done, and a separate worker program
on the laptop (src/retraining/worker.py) asks for the next job, does it,
and reports back:

    DataDrift alert ──> backend creates a "retrain" job (status: queued)
    worker ──claim──>   job becomes "running"   ──> dvc repro (train)
    worker ──finish──>  job becomes "succeeded" or "failed"

    Human clicks Approve ──> backend moves the "champion" alias in MLflow
                             and creates a "deploy" job
    worker ──claim──>   restart model-server, reload backend, check in sync

Jobs are kept in a JSON file in the project folder, so they survive a
backend restart. A lock makes every change all-or-nothing, because two
requests can arrive at the same time.
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

JOB_KINDS = ("retrain", "deploy")
OPEN_STATES = ("queued", "running")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------ file access
    def _read(self) -> list:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())

    def _write(self, jobs: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file, then rename: a crash halfway through
        # writing can never leave a half-written, unreadable jobs file.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------- operations
    def list(self, limit: int = 20) -> list:
        """Newest first."""
        with self._lock:
            return list(reversed(self._read()))[:limit]

    def create(self, kind: str, reason: str, params: dict) -> dict:
        assert kind in JOB_KINDS
        with self._lock:
            jobs = self._read()
            job = {"id": len(jobs) + 1, "kind": kind, "status": "queued", "reason": reason,
                   "params": params, "created_at": now(), "started_at": None,
                   "finished_at": None, "detail": "", "result": {}}
            jobs.append(job)
            self._write(jobs)
            return job

    def open_job(self, kind: str):
        """A job of this kind that is still queued or running, if any."""
        with self._lock:
            return next((j for j in self._read()
                         if j["kind"] == kind and j["status"] in OPEN_STATES), None)

    def created_within(self, kind: str, minutes: float) -> bool:
        """Was a job of this kind created in the last `minutes`? (cooldown)"""
        limit = datetime.now().astimezone() - timedelta(minutes=minutes)
        with self._lock:
            return any(j["kind"] == kind and datetime.fromisoformat(j["created_at"]) > limit
                       for j in self._read())

    def claim_next(self):
        """Oldest queued job becomes "running" and is returned (None if no work)."""
        with self._lock:
            jobs = self._read()
            job = next((j for j in jobs if j["status"] == "queued"), None)
            if job is not None:
                job["status"], job["started_at"] = "running", now()
                self._write(jobs)
            return job

    def finish(self, job_id: int, succeeded: bool, detail: str, result: dict) -> dict:
        """Record the outcome of a running job. Returns the job, or None if unknown."""
        with self._lock:
            jobs = self._read()
            job = next((j for j in jobs if j["id"] == job_id), None)
            if job is None:
                return None
            if job["status"] != "running":
                raise ValueError(f"Job {job_id} is {job['status']}, not running")
            job.update({"status": "succeeded" if succeeded else "failed",
                        "finished_at": now(), "detail": detail, "result": result})
            self._write(jobs)
            return job
