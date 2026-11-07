"""
Prometheus metrics of the backend, served at GET /metrics.

Prometheus "scrapes" (reads) this page every 15 seconds and stores every
value with a timestamp; Grafana then draws graphs from those stored values.

Four kinds of metric are used, each for a different job:
    Counter    only goes up (number of requests, predictions, errors).
               Prometheus turns it into a RATE: requests per second.
    Histogram  counts values into buckets (request times, probabilities),
               so Prometheus can compute percentiles such as the 95th.
    Gauge      a value that goes up and down (is the model loaded? PSI now).
    Info       fixed text labels (which model version is served).

All names start with "pdm_" (predictive maintenance) so they are easy to
find in Prometheus and do not clash with other services' metrics.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    disable_created_metrics,
    generate_latest,
)

# By default every counter also gets a "..._created" timestamp series. We
# never use those, and they double the size of the page, so they are off.
disable_created_metrics()

# --------------------------------------------------------------- HTTP traffic
HTTP_REQUESTS = Counter(
    "pdm_http_requests_total", "HTTP requests handled by the backend", ["method", "path", "status"]
)
HTTP_LATENCY = Histogram(
    "pdm_http_request_duration_seconds",
    "Time to answer an HTTP request",
    ["method", "path"],
    # Buckets in seconds: most requests take a few ms; a CSV with SHAP can take 1-2 s.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# ----------------------------------------------------------------- predictions
PREDICTIONS = Counter("pdm_predictions_total", "Engines predicted, by risk level", ["risk_level"])
FAILURE_PROBABILITY = Histogram(
    "pdm_failure_probability",
    "Predicted failure probability of each engine",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
ENGINES_PER_REQUEST = Histogram(
    "pdm_engines_per_request",
    "Number of engines in one prediction request",
    buckets=(1, 2, 5, 10, 20, 50, 100, 250),
)
ROWS_RECEIVED = Counter(
    "pdm_sensor_rows_received_total", "Sensor reading rows received for prediction"
)
VALIDATION_ERRORS = Counter(
    "pdm_validation_errors_total", "Requests rejected because of bad input", ["error"]
)
MODEL_SERVER_LATENCY = Histogram(
    "pdm_model_server_duration_seconds",
    "Time the MLflow model server takes to answer",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

# --------------------------------------------------------------- model state
MODEL_INFO = Info("pdm_model", "The model version currently served")
MODEL_LOADED = Gauge("pdm_model_loaded", "1 if a model is loaded, else 0")
MODEL_SERVER_UP = Gauge("pdm_model_server_up", "1 if the model server answers, else 0")
MODEL_SERVER_IN_SYNC = Gauge(
    "pdm_model_server_in_sync",
    "1 if the model server serves the same version as the backend, 0 if not, -1 if unknown",
)
MODEL_THRESHOLD = Gauge("pdm_model_threshold", "Decision threshold of the served model")

# ----------------------------------------------------------------- data drift
DRIFT_PSI = Gauge(
    "pdm_drift_psi", "PSI of each sensor: recent readings vs training data", ["sensor"]
)
DRIFT_MAX_PSI = Gauge("pdm_drift_max_psi", "Highest sensor PSI (-1 while not enough rows)")
DRIFT_DETECTED = Gauge("pdm_drift_detected", "1 if max PSI is above the threshold, else 0")
DRIFT_THRESHOLD = Gauge("pdm_drift_threshold", "PSI threshold of the drift alarm")
DRIFT_WINDOW_ROWS = Gauge("pdm_drift_window_rows", "Readings currently in the drift window")

# -------------------------------------------------------------------- alerts
ALERTS_RECEIVED = Counter(
    "pdm_alerts_received_total",
    "Alert notifications received from Alertmanager",
    ["alertname", "status"],
)


# ------------------------------------------------------- retraining (Task 9)
JOBS_CREATED = Counter(
    "pdm_jobs_created_total", "Retrain and deploy jobs created", ["kind", "trigger"]
)
JOBS_FINISHED = Counter("pdm_jobs_finished_total", "Jobs finished, by outcome", ["kind", "status"])
JOBS_OPEN = Gauge("pdm_jobs_open", "Jobs queued or running", ["kind"])


def route_label(request) -> str:
    """
    The path TEMPLATE (e.g. "/predict"), not the raw URL. Unknown URLs are
    grouped as "other": otherwise someone requesting thousands of random
    URLs would create thousands of separate time series in Prometheus.
    """
    route = request.scope.get("route")
    return getattr(route, "path", "other")


def record_predictions(result: dict, n_rows: int) -> None:
    """Called after every successful prediction request."""
    ROWS_RECEIVED.inc(n_rows)
    ENGINES_PER_REQUEST.observe(result["n_engines"])
    for engine in result["engines"]:
        PREDICTIONS.labels(risk_level=engine["risk_level"]).inc()
        FAILURE_PROBABILITY.observe(engine["failure_probability"])


def update_state_gauges(service, drift_status: dict) -> None:
    """
    Gauges describe the state NOW, so they are refreshed just before
    Prometheus reads them (when /metrics is requested), not on every request.
    """
    MODEL_LOADED.set(1 if service.is_loaded else 0)
    if service.is_loaded:
        MODEL_INFO.info(
            {
                "version": str(service.info.get("version")),
                "model_type": str(service.info.get("model_type", "unknown")),
                "git_commit": str(service.info.get("git_commit", "unknown")),
            }
        )
        MODEL_THRESHOLD.set(service.info["threshold"])

    server, in_sync = service.model_server_status()
    MODEL_SERVER_UP.set(1 if server in ("reachable", "not_configured") else 0)
    MODEL_SERVER_IN_SYNC.set(-1 if in_sync is None else int(in_sync))

    DRIFT_THRESHOLD.set(drift_status["threshold"])
    DRIFT_WINDOW_ROWS.set(drift_status["n_rows"])
    DRIFT_DETECTED.set(1 if drift_status["drift_detected"] else 0)
    max_psi = drift_status["max_psi"]
    DRIFT_MAX_PSI.set(-1 if max_psi is None else max_psi)
    DRIFT_PSI.clear()  # no stale per-sensor values while the window is refilling
    for sensor, value in drift_status["sensor_psi"].items():
        DRIFT_PSI.labels(sensor=sensor).set(value)


def metrics_page() -> tuple:
    """The text Prometheus reads, and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
