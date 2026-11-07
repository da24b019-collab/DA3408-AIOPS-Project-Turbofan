"""
Tests for the pipeline information shown on the dashboard's pipeline page.
Each test builds a tiny fake project (dvc.yaml, reports, logs) in a
temporary folder, so it does not depend on the real pipeline's state.
"""

import json

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.pipeline_info import pipeline_status, recent_log_events
from tests.test_api import FakeModelService, make_settings

DVC_YAML = """
stages:
  ingest:
    cmd: python -m src.data.ingest
    deps: [data/raw]
    outs: [data/interim]
    metrics:
      - reports/ingest_report.json:
          cache: false
  features:
    cmd: python -m src.features.build_features
    deps: [data/interim]
    outs: [data/features]
    metrics:
      - reports/features_report.json:
          cache: false
"""


def make_project(tmp_path):
    (tmp_path / "dvc.yaml").write_text(DVC_YAML)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "ingest_report.json").write_text(json.dumps(
        {"subsets": {"FD001": {"train": {"rows": 100, "engines": 2},
                               "test": {"rows": 50, "engines": 2}}}}))
    (tmp_path / "reports" / "features_report.json").write_text(json.dumps(
        {"total_rows": 150, "processing_seconds": 3.0, "rows_per_second": 50, "extra": "x"}))
    return tmp_path


def test_stages_and_arrows(tmp_path):
    result = pipeline_status(make_project(tmp_path), run_dvc_status=False)
    assert [s["name"] for s in result["stages"]] == ["ingest", "features"]
    # raw data feeds ingest; ingest's output (data/interim) feeds features
    assert ["data/raw.dvc", "ingest"] in result["edges"]
    assert ["ingest", "features"] in result["edges"]


def test_stage_metrics_are_extracted(tmp_path):
    stages = {s["name"]: s for s in pipeline_status(make_project(tmp_path),
                                                    run_dvc_status=False)["stages"]}
    assert stages["ingest"]["metrics"] == {"total_rows": 150, "engines": 4}
    assert stages["features"]["metrics"]["rows_per_second"] == 50
    assert "extra" not in stages["features"]["metrics"]   # only the chosen keys
    assert stages["features"]["status"] == "unknown"      # DVC status not run in this test


def test_missing_dvc_yaml(tmp_path):
    assert pipeline_status(tmp_path, run_dvc_status=False) == {
        "stages": [], "edges": [], "dvc_available": False}


def test_log_events_keep_only_warnings_and_errors(tmp_path):
    (tmp_path / "api.log").write_text(
        "2026-11-07 10:00:00,000 | INFO     | api | all fine\n"
        "2026-11-07 10:00:01,000 | WARNING  | api | slow request\n"
        "2026-11-07 10:00:02,000 | ERROR    | api | model server down\n"
        "a line that is not in our log format\n")
    events = recent_log_events(tmp_path, limit=10)
    assert [e["level"] for e in events] == ["ERROR", "WARNING"]   # newest first
    assert events[0]["message"] == "model server down"


def test_runs_endpoint_reports_mlflow_down():
    settings = make_settings()
    settings.mlflow_tracking_uri = "http://127.0.0.1:9"  # nothing listens on port 9
    with TestClient(create_app(settings, FakeModelService(settings))) as client:
        response = client.get("/pipeline/runs")
        assert response.status_code == 503
        assert response.json()["error"] == "service_unavailable"
