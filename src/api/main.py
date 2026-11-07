"""
FastAPI backend: the only part of the system the dashboard talks to.

Endpoints (full specification in docs/lld_api.md):
    GET  /health         is the process up?
    GET  /ready          can it predict? is the model server in sync?
    GET  /model/info     which model is being served
    POST /model/reload   load the current champion (after an approval)
    POST /predict        predict from JSON readings
    POST /predict/csv    predict from an uploaded CSV file
    GET  /pipeline/status   DVC stages, their status and metrics
    GET  /pipeline/runs     MLflow training runs (successful and failed)
    GET  /logs/recent       latest warnings and errors from the logs
    GET  /metrics           Prometheus metrics (read by Prometheus every 15 s)
    GET  /drift/status      data drift (PSI) of recent readings vs training data
    POST /drift/reset       empty the drift window (e.g. after a planned change)
    POST /alerts/webhook    Alertmanager sends fired and resolved alerts here
    GET  /alerts/recent     the alerts received most recently
  Retraining loop (Task 9):
    POST /retrain           queue a retraining job (a DataDrift alert does this too)
    GET  /jobs              retrain and deploy jobs, newest first
    POST /jobs/claim        the worker takes the oldest queued job
    POST /jobs/{id}/finish  the worker reports the outcome
    GET  /model/versions    all registered versions and their aliases
    GET  /model/compare     champion vs challenger on the same test sets
    POST /model/approve     human approval: challenger becomes champion (+ deploy job)
    POST /model/rollback    an earlier version becomes champion (+ deploy job)

Run it with:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Interactive documentation: http://127.0.0.1:8000/docs
"""

import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from src.api.engine_data import (
    BadFileError,
    InputValidationError,
    read_csv_upload,
    readings_to_frame,
    validate_readings,
)
from src.api import metrics
from src.api.jobs import JobStore
from src.api.registry import Registry, RegistryError, VersionNotFound
from src.api.model_service import ModelService, ModelUnavailableError
from src.api.pipeline_info import mlflow_runs, pipeline_status, recent_log_events
from src.api.prediction import predict_engines
from src.api.schemas import (
    AlertmanagerWebhook,
    AlertsResponse,
    ClaimResponse,
    CompareResponse,
    DeployResponse,
    FinishJobRequest,
    JobsResponse,
    RetrainRequest,
    RollbackRequest,
    VersionsResponse,
    DriftStatusResponse,
    ErrorResponse,
    HealthResponse,
    LogEventsResponse,
    ModelInfoResponse,
    PipelineStatusResponse,
    PredictionResponse,
    PredictRequest,
    ReadyResponse,
    RunsResponse,
)
from src.api.settings import Settings
from src.models.tracking import check_mlflow_server
from src.logger import LOG_DIR, get_logger
from src.monitoring.drift import DriftMonitor

logger = get_logger("api", log_file="api.log")

ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 401, 404, 409, 422, 503)}
MAX_STORED_ALERTS = 50


def create_app(
    settings: Settings = None,
    service: ModelService = None,
    registry: Registry = None,
    jobs: JobStore = None,
) -> FastAPI:
    """
    Build the app. Tests pass in their own settings and a fake model service,
    so they run without MLflow; in normal use both are created here.
    """
    settings = settings or Settings.from_env()
    service = service or ModelService(settings)
    drift = DriftMonitor(
        settings.sensors,
        settings.drift_threshold,
        settings.drift_min_rows,
        settings.drift_window_rows,
    )
    received_alerts = deque(maxlen=MAX_STORED_ALERTS)  # newest last
    registry = registry or Registry(settings, settings.max_fd001_drop)
    jobs = jobs or JobStore(settings.jobs_file)

    def queue_retraining(reason: str, subsets: list, trigger: str):
        """
        Create a retrain job unless one is already waiting or running.
        Returns (job, message); job is None when nothing was created.
        """
        if jobs.open_job("retrain"):
            return None, "skipped: a retrain job is already queued or running"
        if trigger == "alert" and jobs.created_within("retrain", settings.retrain_cooldown_minutes):
            # Without a cooldown, an alert that keeps firing would start job after job
            return None, f"skipped: cooldown ({settings.retrain_cooldown_minutes:.0f} min)"
        job = jobs.create("retrain", reason, {"subsets": list(subsets)})
        metrics.JOBS_CREATED.labels("retrain", trigger).inc()
        logger.warning("Retrain job %d queued (%s): %s", job["id"], trigger, reason)
        return job, f"retrain job {job['id']} queued"

    def sync_drift_baseline():
        """
        The drift baseline belongs to the model version. When a different
        version has been loaded (start-up, /model/reload), switch to its baseline.
        """
        version = service.info.get("version")
        baseline = getattr(service, "drift_baseline", None)
        if version != drift.baseline_version and baseline is not None:
            drift.set_baseline(baseline, version)
            logger.info("Drift baseline switched to model version %s", version)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Try to load the model at start-up. If the registry is down, the API
        # still starts (so /health works) and /ready reports not ready.
        if not service.is_loaded:
            try:
                service.load()
            except ModelUnavailableError as err:
                logger.error("Start-up without a model: %s", err)
        yield

    app = FastAPI(
        title="Turbofan Predictive Maintenance API",
        version="1.0.0",
        description="Predicts whether an engine will fail within 30 cycles.",
        lifespan=lifespan,
    )
    app.state.settings, app.state.service = settings, service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def log_and_measure_requests(request: Request, call_next):
        """
        For every request: one log line, plus the Prometheus request counter
        and timer (method, path, status, time taken).
        """
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500  # an unexpected error: still count it, then let it be handled
            raise
        finally:
            seconds = time.perf_counter() - start
            path = metrics.route_label(request)
            metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            metrics.HTTP_LATENCY.labels(request.method, path).observe(seconds)
        logger.info(
            "%s %s -> %d (%.1f ms)", request.method, request.url.path, status, seconds * 1000
        )
        return response

    # -------------------------------------------------------- error handlers
    def error(status: int, code: str, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": code, "detail": detail})

    @app.exception_handler(BadFileError)
    async def bad_file(_, exc):
        metrics.VALIDATION_ERRORS.labels("bad_file").inc()
        return error(400, "bad_file", str(exc))

    @app.exception_handler(InputValidationError)
    async def invalid_input(_, exc):
        metrics.VALIDATION_ERRORS.labels("invalid_input").inc()
        return error(422, "invalid_input", str(exc))

    @app.exception_handler(RequestValidationError)
    async def invalid_request_body(_, exc):
        # Wrong JSON shape (e.g. no "readings" field): same error format as everything else
        metrics.VALIDATION_ERRORS.labels("invalid_request").inc()
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return error(422, "invalid_input", f"Invalid request: {problems}")

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable(_, exc):
        logger.error("Model unavailable: %s", exc)
        return error(503, "model_unavailable", str(exc))

    @app.exception_handler(Exception)
    async def unexpected(_, exc):
        # Full details go to the log; the user gets a generic message.
        logger.exception("Unexpected error: %s", exc)
        return error(500, "internal_error", "Something went wrong. The error has been logged.")

    class ServiceUnavailable(Exception):
        """Another service this endpoint needs (e.g. the MLflow tracking server) is down."""

    @app.exception_handler(ServiceUnavailable)
    async def service_unavailable(_, exc):
        logger.error("Service unavailable: %s", exc)
        return error(503, "service_unavailable", str(exc))

    # ----------------------------------------------------------- API key
    class Conflict(Exception):
        """The request is valid but not allowed in the current state."""

    @app.exception_handler(Conflict)
    async def conflict(_, exc):
        return error(409, "conflict", str(exc))

    class NotFound(Exception):
        pass

    @app.exception_handler(NotFound)
    async def not_found(_, exc):
        return error(404, "not_found", str(exc))

    @app.exception_handler(VersionNotFound)
    async def version_not_found(_, exc):
        return error(404, "not_found", str(exc))

    @app.exception_handler(RegistryError)
    async def registry_error(_, exc):
        logger.error("Registry problem: %s", exc)
        return error(503, "service_unavailable", str(exc))

    class Unauthorized(Exception):
        pass

    @app.exception_handler(Unauthorized)
    async def unauthorized(_, exc):
        return error(401, "unauthorized", "Missing or wrong X-API-Key header.")

    def require_api_key(request: Request):
        """
        The key is accepted in the X-API-Key header (dashboard, curl) or as
        "Authorization: Bearer <key>" (the only form Alertmanager can send).
        """
        if not settings.api_key:
            return
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if settings.api_key not in (request.headers.get("X-API-Key"), bearer):
            raise Unauthorized()

    protected = [Depends(require_api_key)]

    # ------------------------------------------------------------- routes
    @app.get("/health", response_model=HealthResponse, tags=["status"])
    def health():
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["status"])
    def ready():
        service.load_if_missing()
        server, in_sync = service.model_server_status()
        # Ready only if predictions can succeed AND come from the same model version
        # whose threshold and explanations we use (in_sync is None when not checked)
        is_ready = (
            service.is_loaded and server in ("reachable", "not_configured") and in_sync is not False
        )
        body = ReadyResponse(
            ready=is_ready,
            model_loaded=service.is_loaded,
            model_version=service.info.get("version"),
            model_server=server,
            model_server_in_sync=in_sync,
        )
        return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        """
        Open (no API key) like /health: it holds counts and timings only, no
        sensor data, and the port is only published on 127.0.0.1.
        """
        service.load_if_missing()
        sync_drift_baseline()
        metrics.update_state_gauges(service, drift.status())
        for kind in ("retrain", "deploy"):
            metrics.JOBS_OPEN.labels(kind).set(1 if jobs.open_job(kind) else 0)
        body, content_type = metrics.metrics_page()
        return Response(content=body, media_type=content_type)

    @app.get(
        "/model/info",
        response_model=ModelInfoResponse,
        tags=["model"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def model_info():
        if not service.is_loaded:
            raise ModelUnavailableError("No model is loaded.")
        return service.info

    @app.post(
        "/model/reload",
        response_model=ModelInfoResponse,
        tags=["model"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def model_reload():
        previous = service.info.get("version")
        info = service.load()
        logger.info("Model reloaded: version %s -> %s", previous, info["version"])
        if service.model_server_status()[1] is False:
            logger.warning(
                "Model server is NOT serving version %s. Restart it with " "models:/%s@%s",
                info["version"],
                settings.model_name,
                settings.model_alias,
            )
        return {**info, "previous_version": previous}

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def predict(body: PredictRequest):
        readings = validate_readings(
            readings_to_frame(body.readings), settings.sensors, settings.max_rows
        )
        return run_prediction(readings)

    @app.post(
        "/predict/csv",
        response_model=PredictionResponse,
        tags=["prediction"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    async def predict_csv(
        file: UploadFile = File(..., description="CSV with unit, cycle and sensors")
    ):
        content = await file.read()
        frame = read_csv_upload(file.filename, content, settings.max_upload_mb)
        readings = validate_readings(frame, settings.sensors, settings.max_rows)
        return run_prediction(readings)

    @app.get(
        "/drift/status",
        response_model=DriftStatusResponse,
        tags=["monitoring"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def drift_status():
        sync_drift_baseline()
        return drift.status()

    @app.post(
        "/drift/reset",
        response_model=DriftStatusResponse,
        tags=["monitoring"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def drift_reset():
        drift.reset()
        logger.info("Drift window emptied by request")
        return drift.status()

    @app.post(
        "/alerts/webhook", tags=["monitoring"], dependencies=protected, responses=ERROR_RESPONSES
    )
    def alerts_webhook(body: AlertmanagerWebhook):
        """Alertmanager calls this when an alert starts firing and when it resolves."""
        for alert in body.alerts:
            name = alert.labels.get("alertname", "unknown")
            metrics.ALERTS_RECEIVED.labels(name, alert.status).inc()
            action = ""
            # The closed loop: data drift starts a retraining job automatically
            if name == "DataDrift" and alert.status == "firing":
                drift_now = drift.status()
                reason = "DataDrift alert"
                if drift_now["max_psi"] is not None:
                    reason += f": max PSI {drift_now['max_psi']:.2f} on {drift_now['worst_sensor']}"
                _, action = queue_retraining(reason, settings.retrain_subsets, trigger="alert")
            received_alerts.append(
                {
                    "received_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "alertname": name,
                    "status": alert.status,
                    "severity": alert.labels.get("severity", "none"),
                    "summary": alert.annotations.get("summary", ""),
                    "action": action,
                }
            )
            log = logger.warning if alert.status == "firing" else logger.info
            log("Alert %s %s: %s", name, alert.status, alert.annotations.get("summary", ""))
        return {"received": len(body.alerts)}

    @app.get(
        "/alerts/recent",
        response_model=AlertsResponse,
        tags=["monitoring"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def alerts_recent(limit: int = Query(20, ge=1, le=MAX_STORED_ALERTS)):
        return {"alerts": list(reversed(received_alerts))[:limit]}  # newest first

    # ------------------------------------------------------ retraining loop
    @app.post(
        "/retrain",
        response_model=JobsResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def retrain(body: RetrainRequest):
        subsets = body.subsets or list(settings.retrain_subsets)
        job, message = queue_retraining(body.reason, subsets, trigger="manual")
        if job is None:
            raise Conflict(message)
        return {"jobs": [job]}

    @app.get(
        "/jobs",
        response_model=JobsResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def list_jobs(limit: int = Query(20, ge=1, le=200)):
        return {"jobs": jobs.list(limit)}

    @app.post(
        "/jobs/claim",
        response_model=ClaimResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def claim_job():
        """Called by the worker every few seconds. Returns {"job": null} when there is no work."""
        job = jobs.claim_next()
        if job:
            logger.info("Job %d (%s) claimed by the worker", job["id"], job["kind"])
        return {"job": job}

    @app.post(
        "/jobs/{job_id}/finish",
        response_model=JobsResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def finish_job(job_id: int, body: FinishJobRequest):
        try:
            job = jobs.finish(job_id, body.succeeded, body.detail, body.result)
        except ValueError as err:
            raise Conflict(str(err)) from err
        if job is None:
            raise NotFound(f"No job {job_id}")
        metrics.JOBS_FINISHED.labels(job["kind"], job["status"]).inc()
        log = logger.info if body.succeeded else logger.error
        log("Job %d (%s) %s: %s", job_id, job["kind"], job["status"], body.detail)
        return {"jobs": [job]}

    @app.get(
        "/model/versions",
        response_model=VersionsResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def model_versions():
        return {"versions": registry.versions()}

    @app.get(
        "/model/compare",
        response_model=CompareResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def model_compare():
        return registry.compare()

    def deploy(version: str, reason: str) -> dict:
        """Move the champion alias, then let the worker restart the model server."""
        if jobs.open_job("deploy"):
            raise Conflict("A deploy job is already queued or running. Wait for it to finish.")
        registry.set_champion(version)
        job = jobs.create("deploy", reason, {"version": version})
        metrics.JOBS_CREATED.labels("deploy", "manual").inc()
        logger.warning(
            "Champion is now version %s; deploy job %d queued (%s)", version, job["id"], reason
        )
        return {"champion_version": version, "job": job}

    @app.post(
        "/model/approve",
        response_model=DeployResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def model_approve():
        """The human approval step. Refused unless every promotion check passes."""
        result = registry.compare()
        if result["champion_version"] == result["challenger_version"]:
            raise Conflict(f"Version {result['challenger_version']} is already champion.")
        failed = [c["name"] for c in result["checks"] if not c["passed"]]
        if failed:
            raise Conflict("Approval refused, failed checks: " + "; ".join(failed))
        return deploy(
            result["challenger_version"],
            f"Approved: version {result['challenger_version']} replaces "
            f"{result['champion_version']}",
        )

    @app.post(
        "/model/rollback",
        response_model=DeployResponse,
        tags=["retraining"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def model_rollback(body: RollbackRequest):
        """Emergency switch back to an earlier version (no checks: speed matters here)."""
        current = registry.alias_version("champion")
        if body.version == current:
            raise Conflict(f"Version {body.version} is already champion.")
        return deploy(body.version, f"Rollback: version {body.version} replaces {current}")

    @app.get(
        "/pipeline/status",
        response_model=PipelineStatusResponse,
        tags=["pipeline"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def get_pipeline_status():
        return pipeline_status(settings.project_dir)

    @app.get(
        "/pipeline/runs",
        response_model=RunsResponse,
        tags=["pipeline"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def get_pipeline_runs(limit: int = Query(20, ge=1, le=200)):
        try:
            check_mlflow_server(settings.mlflow_tracking_uri)  # fail in seconds if it is down
            return mlflow_runs(
                settings.mlflow_tracking_uri, settings.experiment_name, settings.model_name, limit
            )
        except Exception as err:
            logger.error("Could not read MLflow runs: %s", err)  # full detail in the log only
            raise ServiceUnavailable("The MLflow tracking server is not reachable.") from err

    @app.get(
        "/logs/recent",
        response_model=LogEventsResponse,
        tags=["pipeline"],
        dependencies=protected,
        responses=ERROR_RESPONSES,
    )
    def get_recent_logs(limit: int = Query(50, ge=1, le=500)):
        return {"events": recent_log_events(LOG_DIR, limit)}

    def run_prediction(readings):
        service.load_if_missing()
        if not service.is_loaded:
            raise ModelUnavailableError("No model is loaded.")
        result = predict_engines(readings, service, settings.sensors, settings.window)
        # Monitoring: count the predictions, and add the readings to the drift window
        metrics.record_predictions(result, len(readings))
        sync_drift_baseline()
        drift.add(readings)
        at_risk = sum(e["at_risk"] for e in result["engines"])
        logger.info(
            "Predicted %d engine(s), %d at risk (model v%s)",
            result["n_engines"],
            at_risk,
            result["model_version"],
        )
        return result

    return app


app = create_app()
