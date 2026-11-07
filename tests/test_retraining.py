"""
Tests for the retraining loop (Task 9): job queue, the DataDrift trigger,
the approval gate and rollback. MLflow is replaced by FakeRegistry, so the
rules are tested without any server.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.jobs import JobStore
from src.api.main import create_app
from src.api.registry import VersionNotFound
from tests.test_api import FakeModelService, make_settings

DRIFT_ALERT = {
    "status": "firing",
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "DataDrift", "severity": "warning"},
            "annotations": {"summary": "drift"},
        }
    ],
}


def scores(fd001_recall, fd001_pr_auc):
    other = {"recall": 0.5, "precision": 0.5, "pr_auc": 0.5}
    return {
        "FD001": {"recall": fd001_recall, "precision": 0.8, "pr_auc": fd001_pr_auc},
        "FD002": other,
        "FD003": other,
        "FD004": other,
    }


class FakeRegistry:
    """Champion v7; challenger v8 whose FD001 scores the test chooses."""

    def __init__(self, challenger_recall=0.78, champion="7", challenger="8"):
        self.aliases = {"champion": champion, "challenger": challenger}
        self.challenger_recall = challenger_recall

    def alias_version(self, alias):
        return self.aliases.get(alias)

    def versions(self):
        return [
            {
                "version": v,
                "aliases": [a for a, x in self.aliases.items() if x == v],
                "train_subsets": "FD001",
                "test_recall": "0.800",
                "created": 0,
            }
            for v in ["8", "7", "6"]
        ]

    def compare(self):
        from src.models.compare import promotion_checks

        champ, chall = scores(0.80, 0.938), scores(self.challenger_recall, 0.943)
        checks = promotion_checks(champ, chall, max_drop=0.05)
        return {
            "champion_version": self.aliases["champion"],
            "challenger_version": self.aliases["challenger"],
            "scores": {"champion": champ, "challenger": chall},
            "checks": checks,
            "approvable": all(c["passed"] for c in checks),
        }

    def set_champion(self, version):
        if version not in {"6", "7", "8"}:
            raise VersionNotFound(f"Model version {version} does not exist")
        self.aliases["champion"] = version


@pytest.fixture
def setup(tmp_path):
    settings = make_settings()
    registry = FakeRegistry()
    jobs = JobStore(tmp_path / "jobs.json")
    with TestClient(create_app(settings, FakeModelService(settings), registry, jobs)) as c:
        yield c, registry, jobs


# ------------------------------------------------------------------ job store
def test_job_lifecycle(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.create("retrain", "test", {"subsets": ["FD001"]})
    assert job["status"] == "queued" and store.open_job("retrain")["id"] == 1
    assert store.claim_next()["status"] == "running"
    assert store.claim_next() is None  # nothing else queued
    store.finish(1, True, "ok", {"model_version": "8"})
    assert store.open_job("retrain") is None
    assert JobStore(tmp_path / "jobs.json").list()[0]["status"] == "succeeded"  # saved to disk
    with pytest.raises(ValueError):
        store.finish(1, True, "again", {})  # only a running job can finish


# --------------------------------------------------------------- the trigger
def test_drift_alert_queues_one_retrain_job(setup):
    client, _, jobs = setup
    client.post("/alerts/webhook", json=DRIFT_ALERT)
    client.post("/alerts/webhook", json=DRIFT_ALERT)  # the alert repeats
    all_jobs = client.get("/jobs").json()["jobs"]
    assert len(all_jobs) == 1  # no duplicate job
    assert all_jobs[0]["kind"] == "retrain"
    assert all_jobs[0]["params"]["subsets"] == ["FD001", "FD002"]
    actions = [a["action"] for a in client.get("/alerts/recent").json()["alerts"]]
    assert actions == [
        "skipped: a retrain job is already queued or running",
        "retrain job 1 queued",
    ]


def test_cooldown_after_a_finished_job(setup):
    client, _, jobs = setup
    client.post("/alerts/webhook", json=DRIFT_ALERT)
    client.post("/jobs/claim")
    client.post("/jobs/1/finish", json={"succeeded": True})
    client.post("/alerts/webhook", json=DRIFT_ALERT)  # within 30 minutes
    assert len(client.get("/jobs").json()["jobs"]) == 1
    # A person can still start one by hand
    assert client.post("/retrain", json={"subsets": ["FD001"]}).status_code == 200


def test_resolved_alert_does_not_retrain(setup):
    client, _, _ = setup
    resolved = {
        "status": "resolved",
        "alerts": [{**DRIFT_ALERT["alerts"][0], "status": "resolved"}],
    }
    client.post("/alerts/webhook", json=resolved)
    assert client.get("/jobs").json()["jobs"] == []


def test_worker_claims_and_finishes(setup):
    client, _, _ = setup
    assert client.post("/jobs/claim").json() == {"job": None}
    client.post("/retrain", json={})
    job = client.post("/jobs/claim").json()["job"]
    assert job["status"] == "running"
    done = client.post(
        f"/jobs/{job['id']}/finish", json={"succeeded": False, "detail": "dvc repro failed"}
    ).json()
    assert done["jobs"][0]["status"] == "failed"
    assert client.post("/jobs/99/finish", json={"succeeded": True}).status_code == 404


# ------------------------------------------------------------- approval gate
def test_approve_moves_alias_and_queues_deploy(setup):
    client, registry, _ = setup
    response = client.post("/model/approve")
    assert response.status_code == 200
    assert registry.aliases["champion"] == "8"
    assert response.json()["job"]["kind"] == "deploy"
    assert response.json()["job"]["params"] == {"version": "8"}


def test_approve_refused_when_fd001_gets_worse(tmp_path):
    settings = make_settings()
    registry = FakeRegistry(challenger_recall=0.70)  # 0.80 -> 0.70: too big a drop
    app = create_app(settings, FakeModelService(settings), registry, JobStore(tmp_path / "j.json"))
    with TestClient(app) as client:
        response = client.post("/model/approve")
        assert response.status_code == 409
        assert "FD001 recall" in response.json()["detail"]
        assert registry.aliases["champion"] == "7"  # nothing changed


def test_only_one_deploy_at_a_time(setup):
    client, _, _ = setup
    client.post("/model/approve")
    assert client.post("/model/rollback", json={"version": "6"}).status_code == 409


def test_rollback(setup):
    client, registry, _ = setup
    assert (
        client.post("/model/rollback", json={"version": "7"}).status_code == 409
    )  # already champion
    response = client.post("/model/rollback", json={"version": "6"})
    assert response.status_code == 200 and registry.aliases["champion"] == "6"
    client.post("/jobs/claim")
    client.post("/jobs/1/finish", json={"succeeded": True})
    assert (
        client.post("/model/rollback", json={"version": "42"}).status_code == 404
    )  # unknown version
