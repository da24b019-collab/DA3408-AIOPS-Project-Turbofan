"""
Retraining worker: runs on the laptop and does the jobs the backend queues.

    python -m src.retraining.worker          # keep running, check for work every 10 s
    python -m src.retraining.worker --once   # do at most one job, then exit

Why a separate program on the laptop and not inside the backend container?
Training needs Spark, DVC and the Git repository, and deploying needs
`docker compose`. Giving the backend container control of Docker would mean
that anyone who breaks into the API controls every container. The backend
only keeps a list of jobs; this worker, run by the project's owner, does them.
(This is the same idea as a CI "runner": it pulls work, it is not pushed to.)

Two kinds of job:

  retrain  1. refuse if Git has uncommitted changes (the run must be reproducible)
           2. set train.subsets in params.yaml, and commit that change
           3. dvc repro -f -s train   -> new model version, alias "challenger"
           4. commit dvc.lock and the reports
           The new version is NOT served yet: a person must approve it.
           If training fails, the project is put back exactly as it was
           before the job (git reset to the starting commit, dvc checkout).

  deploy   (created when a person approves, or rolls back)
           1. docker compose restart model-server   (it loads the new champion)
           2. wait until the model server answers
           3. tell the backend to reload, then check both serve the same version
"""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

from src.config import PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__, log_file="retraining.log")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:5001")
POLL_SECONDS = 10


class JobFailed(Exception):
    """A step of the job failed; the message is reported to the backend."""


# ------------------------------------------------------------------ helpers
def read_api_key() -> str:
    """API_KEY from the environment, else from the project's .env file (never printed)."""
    if os.getenv("API_KEY"):
        return os.environ["API_KEY"]
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def call_backend(method: str, path: str, body: dict = None, timeout: float = 120) -> dict:
    """Send a JSON request to the backend and return the JSON answer."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BACKEND_URL + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "X-API-Key": read_api_key()})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def run(command: list) -> str:
    """Run a command in the project folder; fail the job with its last lines if it fails."""
    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        last_lines = "\n".join(output.splitlines()[-5:])
        raise JobFailed(f"`{' '.join(command)}` failed:\n{last_lines}")
    return output


def git_has_changes() -> bool:
    return bool(run(["git", "status", "--porcelain", "--untracked-files=no"]))


def set_train_subsets(params_text: str, subsets: list) -> str:
    """
    Return params.yaml text with `subsets:` of the TRAIN section replaced.
    (The data section has its own `subsets:` line, which must not change.)
    The text is edited line by line so the comments in params.yaml survive.
    """
    lines = params_text.splitlines(keepends=True)
    in_train = False
    for i, line in enumerate(lines):
        if re.match(r"^\S", line):                 # a top-level section starts
            in_train = line.startswith("train:")
        elif in_train and re.match(r"^  subsets:", line):
            lines[i] = f"  subsets: [{', '.join(subsets)}]\n"
            return "".join(lines)
    raise JobFailed("No `subsets:` line found in the train section of params.yaml")


# --------------------------------------------------------------------- jobs
def do_retrain(job: dict) -> tuple:
    subsets = job["params"]["subsets"]
    if git_has_changes():
        raise JobFailed("The project has uncommitted changes. Commit or stash them first, "
                        "so the new model can be traced to an exact commit.")

    start_commit = run(["git", "rev-parse", "HEAD"])
    params_path = PROJECT_ROOT / "params.yaml"
    old_text = params_path.read_text()
    new_text = set_train_subsets(old_text, subsets)
    try:
        if new_text != old_text:
            params_path.write_text(new_text)
            run(["git", "add", "params.yaml"])
            run(["git", "commit", "-m", f"Retrain job {job['id']}: train on {'+'.join(subsets)}"])
        # -f -s train: run the train stage even if DVC sees no change (the point
        # of a retrain request); the earlier stages are already up to date.
        run(["dvc", "repro", "-f", "-s", "train"])
    except JobFailed:
        # DVC deletes a stage's outputs before running it, so a failed run leaves
        # the project half-changed. The tree was clean when we started, so going
        # back to the starting commit loses nothing but this job's own changes.
        run(["git", "reset", "--hard", start_commit])
        run(["dvc", "checkout", "--force"])
        raise

    report = json.loads((PROJECT_ROOT / "reports" / "train_metrics.json").read_text())
    version = str(report["model_version"])
    run(["git", "add", "dvc.lock", "reports"])
    if git_has_changes():
        run(["git", "commit", "-m", f"Retrain job {job['id']}: model version {version}"])

    detail = (f"Version {version} trained on {'+'.join(subsets)} and registered as "
              f"challenger (test recall {report['test_recall']:.3f}). Waiting for approval.")
    return detail, {"model_version": version, "train_subsets": "+".join(subsets),
                    "test_recall": report["test_recall"], "test_pr_auc": report["test_pr_auc"]}


def wait_for_model_server(timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(MODEL_SERVER_URL + "/ping", timeout=3):
                return
        except OSError:
            time.sleep(3)
    raise JobFailed(f"The model server did not come back within {timeout_seconds} s")


def do_deploy(job: dict) -> tuple:
    version = job["params"]["version"]
    run(["docker", "compose", "restart", "model-server"])
    wait_for_model_server()
    call_backend("POST", "/model/reload")

    ready = call_backend("GET", "/ready")
    if ready["model_version"] != version:
        raise JobFailed(f"Backend serves version {ready['model_version']}, expected {version}")
    if ready["model_server_in_sync"] is False:
        raise JobFailed("The model server and the backend serve different models")
    return (f"Version {version} is live: model server restarted, backend reloaded, "
            f"both in sync."), {"model_version": version}


HANDLERS = {"retrain": do_retrain, "deploy": do_deploy}


def work_once() -> bool:
    """Claim and do one job. Returns False if there was no work."""
    job = call_backend("POST", "/jobs/claim")["job"]
    if job is None:
        return False
    logger.info("Job %d (%s) started: %s", job["id"], job["kind"], job["reason"])
    started = time.perf_counter()
    try:
        detail, result = HANDLERS[job["kind"]](job)
        succeeded = True
    except Exception as err:   # any failure is reported, and the worker keeps running
        detail, result, succeeded = str(err), {}, False
    result["seconds"] = round(time.perf_counter() - started, 1)
    call_backend("POST", f"/jobs/{job['id']}/finish",
                 {"succeeded": succeeded, "detail": detail, "result": result})
    (logger.info if succeeded else logger.error)(
        "Job %d %s in %.0f s: %s", job["id"], "succeeded" if succeeded else "FAILED",
        result["seconds"], detail)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Do the retrain and deploy jobs queued in the backend")
    parser.add_argument("--once", action="store_true", help="do at most one job, then exit")
    args = parser.parse_args()

    logger.info("Worker started: backend %s, checking every %d s (Ctrl+C to stop)",
                BACKEND_URL, POLL_SECONDS)
    while True:
        try:
            did_work = work_once()
        except urllib.error.HTTPError as err:   # the backend answered, but with an error
            hint = " (is API_KEY in .env the same as the backend's?)" if err.code == 401 else ""
            logger.error("Backend refused the request: HTTP %d%s", err.code, hint)
            did_work = False
        except (urllib.error.URLError, OSError) as err:
            logger.warning("Backend not reachable (%s); trying again in %d s", err, POLL_SECONDS)
            did_work = False
        if args.once:
            return
        if not did_work:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Worker stopped")
