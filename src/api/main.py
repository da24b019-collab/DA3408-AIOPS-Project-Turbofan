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

Run it with:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Interactive documentation: http://127.0.0.1:8000/docs
"""

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.engine_data import (
    BadFileError,
    InputValidationError,
    read_csv_upload,
    readings_to_frame,
    validate_readings,
)
from src.api.model_service import ModelService, ModelUnavailableError
from src.api.pipeline_info import mlflow_runs, pipeline_status, recent_log_events
from src.api.prediction import predict_engines
from src.api.schemas import (
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

logger = get_logger("api", log_file="api.log")

ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 401, 422, 503)}


def create_app(settings: Settings = None, service: ModelService = None) -> FastAPI:
    """
    Build the app. Tests pass in their own settings and a fake model service,
    so they run without MLflow; in normal use both are created here.
    """
    settings = settings or Settings.from_env()
    service = service or ModelService(settings)

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

    app = FastAPI(title="Turbofan Predictive Maintenance API", version="1.0.0",
                  description="Predicts whether an engine will fail within 30 cycles.",
                  lifespan=lifespan)
    app.state.settings, app.state.service = settings, service

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """One log line per request: method, path, status and time taken."""
        start = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - start) * 1000)
        return response

    # -------------------------------------------------------- error handlers
    def error(status: int, code: str, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": code, "detail": detail})

    @app.exception_handler(BadFileError)
    async def bad_file(_, exc):
        return error(400, "bad_file", str(exc))

    @app.exception_handler(InputValidationError)
    async def invalid_input(_, exc):
        return error(422, "invalid_input", str(exc))

    @app.exception_handler(RequestValidationError)
    async def invalid_request_body(_, exc):
        # Wrong JSON shape (e.g. no "readings" field): same error format as everything else
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                             for e in exc.errors())
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
    class Unauthorized(Exception):
        pass

    @app.exception_handler(Unauthorized)
    async def unauthorized(_, exc):
        return error(401, "unauthorized", "Missing or wrong X-API-Key header.")

    def require_api_key(request: Request):
        if settings.api_key and request.headers.get("X-API-Key") != settings.api_key:
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
        is_ready = (service.is_loaded and server in ("reachable", "not_configured")
                    and in_sync is not False)
        body = ReadyResponse(ready=is_ready, model_loaded=service.is_loaded,
                             model_version=service.info.get("version"),
                             model_server=server, model_server_in_sync=in_sync)
        return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["model"],
             dependencies=protected, responses=ERROR_RESPONSES)
    def model_info():
        if not service.is_loaded:
            raise ModelUnavailableError("No model is loaded.")
        return service.info

    @app.post("/model/reload", response_model=ModelInfoResponse, tags=["model"],
              dependencies=protected, responses=ERROR_RESPONSES)
    def model_reload():
        previous = service.info.get("version")
        info = service.load()
        logger.info("Model reloaded: version %s -> %s", previous, info["version"])
        if service.model_server_status()[1] is False:
            logger.warning("Model server is NOT serving version %s. Restart it with "
                           "models:/%s@%s", info["version"], settings.model_name,
                           settings.model_alias)
        return {**info, "previous_version": previous}

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
              dependencies=protected, responses=ERROR_RESPONSES)
    def predict(body: PredictRequest):
        readings = validate_readings(readings_to_frame(body.readings),
                                     settings.sensors, settings.max_rows)
        return run_prediction(readings)

    @app.post("/predict/csv", response_model=PredictionResponse, tags=["prediction"],
              dependencies=protected, responses=ERROR_RESPONSES)
    async def predict_csv(file: UploadFile = File(..., description="CSV with unit, cycle and sensors")):
        content = await file.read()
        frame = read_csv_upload(file.filename, content, settings.max_upload_mb)
        readings = validate_readings(frame, settings.sensors, settings.max_rows)
        return run_prediction(readings)

    @app.get("/pipeline/status", response_model=PipelineStatusResponse, tags=["pipeline"],
             dependencies=protected, responses=ERROR_RESPONSES)
    def get_pipeline_status():
        return pipeline_status(settings.project_dir)

    @app.get("/pipeline/runs", response_model=RunsResponse, tags=["pipeline"],
             dependencies=protected, responses=ERROR_RESPONSES)
    def get_pipeline_runs(limit: int = Query(20, ge=1, le=200)):
        try:
            check_mlflow_server(settings.mlflow_tracking_uri)  # fail in seconds if it is down
            return mlflow_runs(settings.mlflow_tracking_uri, settings.experiment_name,
                               settings.model_name, limit)
        except Exception as err:
            logger.error("Could not read MLflow runs: %s", err)  # full detail in the log only
            raise ServiceUnavailable("The MLflow tracking server is not reachable.") from err

    @app.get("/logs/recent", response_model=LogEventsResponse, tags=["pipeline"],
             dependencies=protected, responses=ERROR_RESPONSES)
    def get_recent_logs(limit: int = Query(50, ge=1, le=500)):
        return {"events": recent_log_events(LOG_DIR, limit)}

    def run_prediction(readings):
        service.load_if_missing()
        if not service.is_loaded:
            raise ModelUnavailableError("No model is loaded.")
        result = predict_engines(readings, service, settings.sensors, settings.window)
        at_risk = sum(e["at_risk"] for e in result["engines"])
        logger.info("Predicted %d engine(s), %d at risk (model v%s)",
                    result["n_engines"], at_risk, result["model_version"])
        return result

    return app


app = create_app()
