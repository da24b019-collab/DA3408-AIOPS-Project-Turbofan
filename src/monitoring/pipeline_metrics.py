"""
Send pipeline metrics to the Prometheus Pushgateway.

Why a Pushgateway? Prometheus PULLS metrics: it visits each service's
/metrics page every 15 seconds. That works for the backend, which runs all
the time. A pipeline stage (feature building, training) runs for a minute
and then exits, so there is nothing left to visit. The stage therefore
PUSHES its numbers to the Pushgateway at the end, the Pushgateway keeps
them, and Prometheus pulls them from there.

Each stage pushes under its own "job" name, so a new run of a stage
replaces that stage's old values instead of mixing with them.

If the Pushgateway is not running, the stage logs a warning and carries
on: monitoring being down must never make the pipeline fail.
"""

import os
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.config import load_params
from src.logger import get_logger

logger = get_logger(__name__)


def pushgateway_url() -> str:
    """PUSHGATEWAY_URL from the environment, else params.yaml."""
    return os.getenv("PUSHGATEWAY_URL", load_params()["monitoring"]["pushgateway_url"])


def push_stage_metrics(stage: str, duration_seconds: float, success: bool,
                       values: dict = None) -> bool:
    """
    Push one stage's results, for example:
        push_stage_metrics("train", 52.3, True, {"test_recall": 0.94})
    becomes, in Prometheus:
        pdm_pipeline_stage_duration_seconds{job="pdm_train"}  52.3
        pdm_pipeline_stage_success{job="pdm_train"}           1
        pdm_pipeline_stage_last_run_timestamp{job="pdm_train"} <unix time>
        pdm_train_test_recall{job="pdm_train"}                 0.94
    Returns True if the push worked.
    """
    registry = CollectorRegistry()   # a fresh registry: only this stage's values are sent
    Gauge("pdm_pipeline_stage_duration_seconds", "How long the stage ran",
          registry=registry).set(duration_seconds)
    Gauge("pdm_pipeline_stage_success", "1 if the stage finished without error",
          registry=registry).set(1 if success else 0)
    Gauge("pdm_pipeline_stage_last_run_timestamp", "When the stage last finished (unix time)",
          registry=registry).set(time.time())
    for name, value in (values or {}).items():
        Gauge(f"pdm_{stage}_{name}", f"{stage} stage: {name}", registry=registry).set(float(value))

    url = pushgateway_url()
    try:
        push_to_gateway(url, job=f"pdm_{stage}", registry=registry, timeout=5)
        logger.info("Pushed %s metrics to the Pushgateway at %s", stage, url)
        return True
    except OSError as err:   # connection refused, timeout, DNS failure
        logger.warning("Pushgateway not reachable at %s (%s): metrics not pushed", url, err)
        return False
