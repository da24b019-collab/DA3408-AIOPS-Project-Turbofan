"""
Helpers that record WHERE a result came from, so every MLflow run can be
reproduced: the exact Git commit of the code and the exact DVC version of
the data. The guidelines require every experiment to be reproducible from
a Git commit hash plus an MLflow run ID.
"""

import subprocess
import urllib.request

import yaml

from src.config import PROJECT_ROOT


def git_commit() -> str:
    """Full hash of the current Git commit, or 'unknown' outside a Git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def git_is_dirty() -> bool:
    """
    True if there are uncommitted changes. A run made with uncommitted code
    cannot be reproduced exactly from its commit hash, so we record this.
    """
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except subprocess.CalledProcessError, FileNotFoundError:
        return True


def dvc_output_hash(stage: str, path: str) -> str:
    """
    The MD5 hash DVC recorded in dvc.lock for one output of one stage.
    It identifies the exact version of the data, like a commit hash for data.
    """
    lock_file = PROJECT_ROOT / "dvc.lock"
    if not lock_file.exists():
        return "unknown"
    lock = yaml.safe_load(lock_file.read_text())
    for out in lock.get("stages", {}).get(stage, {}).get("outs", []):
        if out.get("path") == path:
            return out.get("md5", "unknown")
    return "unknown"


def check_mlflow_server(tracking_uri: str) -> None:
    """Fail early with a clear message if the MLflow server is not running."""
    try:
        with urllib.request.urlopen(f"{tracking_uri}/health", timeout=5) as response:
            if response.status == 200:
                return
    except OSError:
        pass
    raise RuntimeError(
        f"MLflow server is not reachable at {tracking_uri}. Start it first with:\n"
        "  mlflow server --backend-store-uri sqlite:///mlflow_data/mlflow.db "
        "--artifacts-destination mlflow_data/artifacts --host 127.0.0.1 --port 5000"
    )
