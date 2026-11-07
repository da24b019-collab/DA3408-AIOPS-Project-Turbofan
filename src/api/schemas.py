"""
Request and response shapes (docs/lld_api.md, sections 3 and 4).

FastAPI uses these Pydantic models to check incoming JSON, to shape the
outgoing JSON, and to generate the interactive documentation at /docs, so
the documentation always matches the code.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_version: Optional[str] = None
    model_server: str
    model_server_in_sync: Optional[bool] = None


class SensorInfo(BaseModel):
    code: str
    description: str


class ModelInfoResponse(BaseModel):
    model_name: str
    alias: str
    version: str
    run_id: str
    model_type: str
    threshold: float
    window: int
    n_features: int
    git_commit: str
    test_metrics: dict
    sensors: list[SensorInfo]
    loaded_at: str
    drift_baseline: bool = False  # does this model version have a drift baseline?
    previous_version: Optional[str] = None


class PredictRequest(BaseModel):
    readings: list[dict] = Field(
        ...,
        description="One object per engine per cycle: unit, cycle and the 14 model sensors.",
        examples=[
            [
                {
                    "unit": 1,
                    "cycle": 1,
                    "s2": 642.1,
                    "s3": 1585.3,
                    "s4": 1398.2,
                    "s7": 554.1,
                    "s8": 2388.0,
                    "s9": 9050.2,
                    "s11": 47.3,
                    "s12": 521.9,
                    "s13": 2388.0,
                    "s14": 8133.5,
                    "s15": 8.42,
                    "s17": 392,
                    "s20": 38.9,
                    "s21": 23.3,
                }
            ]
        ],
    )


class Factor(BaseModel):
    sensor: str
    description: str
    effect: Literal["raises risk", "lowers risk"]
    contribution: float


class Explanation(BaseModel):
    summary: str
    top_factors: list[Factor]


class EnginePrediction(BaseModel):
    unit: int
    cycles_received: int
    last_cycle: int
    failure_probability: float
    at_risk: bool
    risk_level: Literal["low", "medium", "high"]
    explanation: Explanation
    warnings: list[str]


class PredictionResponse(BaseModel):
    model_version: str
    threshold: float
    n_engines: int
    engines: list[EnginePrediction]
    warnings: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str


class StageInfo(BaseModel):
    name: str
    command: str
    status: Literal["up_to_date", "changed", "unknown"]
    last_run: Optional[str] = None
    metrics: dict


class PipelineStatusResponse(BaseModel):
    stages: list[StageInfo]
    edges: list[list[str]]
    dvc_available: bool


class RunInfo(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    status: str
    start_time: str
    duration_seconds: Optional[float] = None
    model_type: Optional[str] = None
    git_commit: Optional[str] = None
    test_recall: Optional[float] = None
    test_pr_auc: Optional[float] = None
    model_version: Optional[str] = None
    aliases: list[str] = []


class RunsResponse(BaseModel):
    experiment: str
    runs: list[RunInfo]


class LogEvent(BaseModel):
    time: str
    level: str
    source: str
    message: str
    file: str


class LogEventsResponse(BaseModel):
    events: list[LogEvent]


class DriftStatusResponse(BaseModel):
    """Data drift of the recent readings vs the model's training data (PSI)."""

    state: Literal["no_baseline", "collecting", "ok", "drift"]
    n_rows: int  # readings in the window now
    min_rows: int  # PSI is only judged from this many rows on
    window_rows: int  # size of the sliding window
    threshold: float
    baseline_model_version: Optional[str] = None
    max_psi: Optional[float] = None
    worst_sensor: Optional[str] = None
    drift_detected: bool
    closest_baseline: Optional[str] = None  # training dataset the data looks most like
    baselines: list[str] = []  # datasets the served model was trained on
    sensor_psi: dict[str, float]


class Alert(BaseModel):
    """One alert as Alertmanager sends it (only the fields we use)."""

    status: str  # "firing" or "resolved"
    labels: dict[str, str] = {}
    annotations: dict[str, str] = {}
    startsAt: Optional[str] = None
    endsAt: Optional[str] = None


class AlertmanagerWebhook(BaseModel):
    """Body of an Alertmanager webhook notification."""

    status: str
    alerts: list[Alert]


class ReceivedAlert(BaseModel):
    received_at: str
    alertname: str
    status: str
    severity: str
    summary: str
    action: str = ""  # what the backend did about it (e.g. "retrain job 3 queued")


class AlertsResponse(BaseModel):
    alerts: list[ReceivedAlert]


# ------------------------------------------------------------ retraining (Task 9)
class Job(BaseModel):
    """A retraining or deploy job (src/api/jobs.py)."""

    id: int
    kind: Literal["retrain", "deploy"]
    status: Literal["queued", "running", "succeeded", "failed"]
    reason: str
    params: dict
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    detail: str = ""
    result: dict = {}


class JobsResponse(BaseModel):
    jobs: list[Job]


class ClaimResponse(BaseModel):
    job: Optional[Job] = None  # null when there is no work


class RetrainRequest(BaseModel):
    subsets: Optional[list[Literal["FD001", "FD002", "FD003", "FD004"]]] = (
        None  # default: params.yaml
    )
    reason: str = "Requested from the dashboard"


class FinishJobRequest(BaseModel):
    succeeded: bool
    detail: str = ""
    result: dict = {}


class Check(BaseModel):
    name: str
    passed: bool
    detail: str


class CompareResponse(BaseModel):
    """Champion and challenger scored on the same NASA test sets."""

    champion_version: str
    challenger_version: str
    scores: dict[str, dict[str, dict[str, float]]]  # role -> subset -> metric -> value
    checks: list[Check]
    approvable: bool


class RollbackRequest(BaseModel):
    version: str


class DeployResponse(BaseModel):
    champion_version: str
    job: Job


class ModelVersion(BaseModel):
    version: str
    aliases: list[str]
    train_subsets: str
    test_recall: Optional[str] = None
    created: int


class VersionsResponse(BaseModel):
    versions: list[ModelVersion]
