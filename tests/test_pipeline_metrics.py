"""The pipeline must keep working when the monitoring stack is down."""

from src.monitoring.pipeline_metrics import push_stage_metrics


def test_push_failure_does_not_raise(monkeypatch):
    # Port 9 ("discard") on this machine: nothing listens there, so the push fails
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://127.0.0.1:9")
    assert push_stage_metrics("train", 1.0, True, {"test_recall": 0.9}) is False
