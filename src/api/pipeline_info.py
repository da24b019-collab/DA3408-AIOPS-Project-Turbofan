"""
Information about the ML pipeline, for the dashboard's pipeline page.

The frontend must only talk to the backend through REST, so the backend
collects this information from the tools themselves:
    - DVC:    the stages in dvc.yaml, how they connect, and whether each one
              is up to date (`dvc status`)
    - Reports: the metrics each stage wrote (rows, seconds, rows per second)
    - MLflow: the history of training runs, including failed ones
    - Logs:   recent warnings and errors from logs/*.log
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

# Which numbers to show for each stage, taken from its metrics file
STAGE_METRIC_KEYS = {
    "ingest": [],  # nested per subset; summarised separately below
    "features": ["total_rows", "processing_seconds", "rows_per_second", "spark_startup_seconds"],
    "train": ["model_type", "model_version", "train_seconds", "test_recall",
              "test_precision", "test_pr_auc", "threshold"],
}
LOG_LINE = re.compile(r"^(\S+ \S+) \| (\w+)\s*\| (\S+) \| (.*)$")
_status_cache = {"time": 0.0, "value": None}
STATUS_CACHE_SECONDS = 30


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _summarise_ingest(report: dict) -> dict:
    """Total rows and engines over all subsets (the ingest report is nested)."""
    rows = engines = 0
    for subset in report.get("subsets", {}).values():
        for split in subset.values():
            rows += split.get("rows", 0)
            engines += split.get("engines", 0)
    return {"total_rows": rows, "engines": engines}


def dvc_changed_stages(project_root: Path) -> dict | None:
    """
    Stages DVC considers out of date, from `dvc status --json`.
    Returns None if DVC cannot be run (e.g. not installed in this container).
    Cached for 30 seconds because `dvc status` checks file hashes and takes a moment.
    """
    now = time.time()
    if _status_cache["value"] is not None and now - _status_cache["time"] < STATUS_CACHE_SECONDS:
        return _status_cache["value"]
    try:
        result = subprocess.run(["dvc", "status", "--json"], cwd=project_root,
                                capture_output=True, text=True, timeout=60)
        changed = json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    _status_cache.update(time=now, value=changed)
    return changed


def pipeline_status(project_root: Path, run_dvc_status: bool = True) -> dict:
    """Stages, the arrows between them, their status and their latest metrics."""
    dvc_file = project_root / "dvc.yaml"
    if not dvc_file.exists():
        return {"stages": [], "edges": [], "dvc_available": False}
    stages_cfg = yaml.safe_load(dvc_file.read_text()).get("stages", {})
    changed = dvc_changed_stages(project_root) if run_dvc_status else None

    # An arrow A -> B when one of A's outputs is one of B's inputs
    outputs = {name: [str(o) for o in cfg.get("outs", [])] for name, cfg in stages_cfg.items()}
    edges = []
    for name, cfg in stages_cfg.items():
        for dep in cfg.get("deps", []):
            for other, outs in outputs.items():
                if other != name and dep in outs and [other, name] not in edges:
                    edges.append([other, name])
            if dep.startswith("data/raw") and ["data/raw.dvc", name] not in edges:
                edges.append(["data/raw.dvc", name])

    stages = []
    for name, cfg in stages_cfg.items():
        metrics_files = [next(iter(m)) if isinstance(m, dict) else m
                         for m in cfg.get("metrics", [])]
        report_path = project_root / metrics_files[0] if metrics_files else None
        report = _read_json(report_path) if report_path else None
        if report is None:
            metrics = {}
        elif name == "ingest":
            metrics = _summarise_ingest(report)
        else:
            metrics = {k: report[k] for k in STAGE_METRIC_KEYS.get(name, []) if k in report}

        if changed is None:
            status = "unknown"
        else:
            status = "changed" if name in changed else "up_to_date"
        last_run = (datetime.fromtimestamp(report_path.stat().st_mtime).astimezone()
                    .isoformat(timespec="seconds") if report_path and report_path.exists() else None)
        stages.append({"name": name, "command": cfg.get("cmd", ""), "status": status,
                       "last_run": last_run, "metrics": metrics})

    return {"stages": stages, "edges": edges, "dvc_available": changed is not None}


def mlflow_runs(tracking_uri: str, experiment_name: str, model_name: str, limit: int) -> dict:
    """Recent training runs from MLflow, newest first, with their registry version and aliases."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {"experiment": experiment_name, "runs": []}

    # run_id -> (version, aliases) from the model registry.
    # Version search does not return aliases, so we read them from the model itself.
    alias_of_version = {}
    try:
        for alias, version in client.get_registered_model(model_name).aliases.items():
            alias_of_version.setdefault(str(version), []).append(alias)
    except Exception:  # model not registered yet
        pass
    versions = {}
    for mv in client.search_model_versions(f"name='{model_name}'"):
        versions[mv.run_id] = (str(mv.version), sorted(alias_of_version.get(str(mv.version), [])))

    runs = []
    for run in client.search_runs([experiment.experiment_id], order_by=["start_time DESC"],
                                  max_results=limit):
        info, data = run.info, run.data
        duration = ((info.end_time - info.start_time) / 1000
                    if info.end_time and info.start_time else None)
        version, aliases = versions.get(info.run_id, (None, []))
        runs.append({
            "run_id": info.run_id,
            "run_name": info.run_name,
            "status": info.status,  # FINISHED, FAILED, RUNNING or KILLED
            "start_time": datetime.fromtimestamp(info.start_time / 1000).astimezone()
                          .isoformat(timespec="seconds"),
            "duration_seconds": round(duration, 1) if duration is not None else None,
            "model_type": data.tags.get("model_type"),
            "git_commit": data.tags.get("git_commit", "")[:10],
            "test_recall": data.metrics.get("test_recall"),
            "test_pr_auc": data.metrics.get("test_pr_auc"),
            "model_version": version,
            "aliases": aliases,
        })
    return {"experiment": experiment_name, "runs": runs}


def recent_log_events(log_dir: Path, limit: int, levels=("WARNING", "ERROR", "CRITICAL")) -> list:
    """The latest warning and error lines from all our log files, newest first."""
    events = []
    for log_file in sorted(log_dir.glob("*.log")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()[-2000:]
        except OSError:
            continue
        for line in lines:
            match = LOG_LINE.match(line)
            if match and match.group(2) in levels:
                events.append({"time": match.group(1), "level": match.group(2),
                               "source": match.group(3), "message": match.group(4),
                               "file": log_file.name})
    events.sort(key=lambda e: e["time"], reverse=True)
    return events[:limit]
