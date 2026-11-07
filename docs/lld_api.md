# Low-Level Design: Backend API

## 1. Components

| Component | Port | Role |
|---|---|---|
| MLflow model server (`mlflow models serve`) | 5001 | Inference engine. Turns 42 feature values into `[P(safe), P(at risk)]`. Serves the registry version with alias `champion`. |
| FastAPI backend (`src/api`) | 8000 | Business layer used by the frontend. Validates input, computes features, calls the model server, applies the threshold, explains predictions, reports model information. |
| MLflow tracking server | 5000 | Model registry. The backend reads the champion's version, threshold, feature list and drift baseline from it. |
| Prometheus | 9090 | Reads the backend's `/metrics` every 15 s and evaluates the alert rules. |
| Alertmanager | 9093 | Sends fired and resolved alerts to the backend's `/alerts/webhook`. |

The frontend talks **only** to the FastAPI backend, through REST calls to a configurable base URL.
The backend talks to the model server through a configurable URL (`MODEL_SERVER_URL`).

## 2. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` (params.yaml) | Registry location |
| `MODEL_NAME` | `turbofan-failure-classifier` (params.yaml) | Registered model name |
| `MODEL_ALIAS` | `champion` | Which registry alias to serve |
| `MODEL_SERVER_URL` | `http://127.0.0.1:5001` | MLflow model server. Empty = predict with the backend's own copy (development only) |
| `API_KEY` | empty | When set, protected endpoints require header `X-API-Key` with this value |
| `CORS_ORIGINS` | `http://localhost:8501,http://localhost:3000` | Comma-separated origins allowed to call the API from a browser (the dashboard) |
| `MAX_UPLOAD_MB` | `5` | Largest accepted CSV |
| `MAX_ROWS` | `50000` | Largest accepted number of readings per request |
| `PROJECT_DIR` | the project root | Folder holding `dvc.yaml`, `reports/` and the DVC data, for `/pipeline/status` (in Docker: `/project`) |
| `LOG_DIR` | `<project>/logs` | Where log files are written and read by `/logs/recent` |
| `EXPERIMENT_NAME` | `turbofan-failure-prediction` (params.yaml) | MLflow experiment shown by `/pipeline/runs` |

Drift settings come from params.yaml, section `monitoring`: threshold `1.0` (max sensor PSI),
`min_rows` `3000`, `window_rows` `5000`. The reasons are in `docs/monitoring_findings.md`.

## 3. Endpoints

All responses are JSON except `/metrics`. Protected endpoints (marked 🔒) need the key when
`API_KEY` is set, either as the header `X-API-Key: <key>` or as `Authorization: Bearer <key>`
(the only form Alertmanager can send).

### 3.1 `GET /health`
Liveness: is the process up? Never touches the model.
- **200** `{"status": "ok"}`

### 3.2 `GET /ready`
Readiness: can the service make predictions?
- **200**
  ```json
  {"ready": true, "model_loaded": true, "model_version": "6",
   "model_server": "reachable", "model_server_in_sync": true}
  ```
  `model_server`: `reachable` | `unreachable` | `not_configured` | `unknown` (no model loaded yet).
  `model_server_in_sync`: the model server and the backend's copy give identical
  probabilities on a fixed check input (`null` if the server is not reachable).
- **503** same body with `"ready": false` when no model is loaded, when the model server
  is configured but unreachable (predictions would fail), or when it is out of sync
  (its predictions would not match the threshold and explanations of the loaded version).
  After approving a new champion: restart the model server, then call `/model/reload`.
- **Self-healing:** if no model is loaded (e.g. the registry was down at start-up), `/ready`
  and the prediction endpoints retry loading the champion, at most once every 30 seconds.
  The registry is checked first with a 5-second limit, so the API never hangs on a dead registry.

### 3.3 `GET /model/info` 🔒
- **200**
  ```json
  {"model_name": "turbofan-failure-classifier", "alias": "champion", "version": "6",
   "run_id": "…", "model_type": "xgboost", "threshold": 0.5308, "window": 10,
   "n_features": 42, "git_commit": "0b9de917…",
   "test_metrics": {"recall": 0.8, "pr_auc": 0.938},
   "sensors": [{"code": "s2", "description": "LPC outlet temperature (T24)"}, …],
   "loaded_at": "2026-11-07T12:00:00+05:30"}
  ```
- **503** no model loaded.

### 3.4 `POST /model/reload` 🔒
Loads whichever version currently holds the alias (after an approval or rollback).
- **200** the `/model/info` body plus `"previous_version"`.
- **503** registry unreachable or alias missing (the previously loaded model stays in use).

### 3.5 `POST /predict` 🔒
Predict from JSON readings.
- **Request body**
  ```json
  {"readings": [{"unit": 1, "cycle": 1, "s2": 642.1, "s3": 1585.3, …}, …]}
  ```
  One object per engine per cycle. Required keys: `unit`, `cycle` and the 14 model sensors
  (`s2 s3 s4 s7 s8 s9 s11 s12 s13 s14 s15 s17 s20 s21`). Other keys are ignored.
- **200** `PredictionResponse` (section 4).
- **422** validation error (section 5). **503** model or model server unavailable.

### 3.6 `POST /predict/csv` 🔒
Predict from an uploaded CSV file (what the dashboard uses).
- **Request**: `multipart/form-data` with field `file` (a `.csv`, at most `MAX_UPLOAD_MB`).
  Header row required; same columns as 3.5.
- **200** `PredictionResponse`. **400** not a CSV, empty or too large. **422** validation error. **503** as above.

### 3.7 `GET /pipeline/status` 🔒
The DVC pipeline for the dashboard's pipeline page.
- **200**
  ```json
  {"stages": [{"name": "features", "command": "python -m src.features.build_features",
               "status": "up_to_date", "last_run": "2026-11-07T10:30:48+05:30",
               "metrics": {"total_rows": 265256, "processing_seconds": 17.6,
                           "rows_per_second": 15075, "spark_startup_seconds": 3.7}}, …],
   "edges": [["data/raw.dvc", "ingest"], ["ingest", "features"], ["features", "train"]],
   "dvc_available": true}
  ```
  `status`: `up_to_date` | `changed` (inputs changed since the last run, from `dvc status`) |
  `unknown` (DVC not available). `dvc status` results are cached for 30 seconds.

### 3.8 `GET /pipeline/runs?limit=20` 🔒
MLflow training runs, newest first, successful and failed (`limit` 1 to 200).
- **200**
  ```json
  {"experiment": "turbofan-failure-prediction",
   "runs": [{"run_id": "…", "run_name": "xgboost-FD001", "status": "FINISHED",
             "start_time": "2026-11-07T11:16:30+05:30", "duration_seconds": 31.2,
             "model_type": "xgboost", "git_commit": "0b9de917d3", "test_recall": 0.8,
             "test_pr_auc": 0.938, "model_version": "6", "aliases": ["champion"]}, …]}
  ```
  `status`: `FINISHED` | `FAILED` | `RUNNING` | `KILLED`.
- **503** `service_unavailable` when the MLflow tracking server cannot be reached.

### 3.9 `GET /logs/recent?limit=50` 🔒
Latest WARNING, ERROR and CRITICAL lines from `logs/*.log`, newest first (`limit` 1 to 500).
- **200** `{"events": [{"time": "2026-11-07 11:40:02,118", "level": "ERROR", "source": "api",
  "message": "Model unavailable: …", "file": "api.log"}, …]}`

### 3.10 `GET /metrics`
Prometheus text format, open like `/health` (counts and timings only, no sensor data).
Not listed in `/docs`. Refreshes the state gauges (model loaded, model server in sync, drift) on each read.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `pdm_http_requests_total` | counter | method, path, status | Requests handled (path = route template; unknown URLs = `other`) |
| `pdm_http_request_duration_seconds` | histogram | method, path | Time to answer |
| `pdm_predictions_total` | counter | risk_level | Engines predicted |
| `pdm_failure_probability` | histogram | | Predicted probability per engine |
| `pdm_engines_per_request` | histogram | | Engines in one request |
| `pdm_sensor_rows_received_total` | counter | | Reading rows received for prediction |
| `pdm_validation_errors_total` | counter | error | Rejected requests (`bad_file`, `invalid_input`, `invalid_request`) |
| `pdm_model_server_duration_seconds` | histogram | | Time the MLflow model server takes |
| `pdm_model_info` | info | version, model_type, git_commit | Served model |
| `pdm_model_loaded`, `pdm_model_server_up` | gauge | | 1 or 0 |
| `pdm_model_server_in_sync` | gauge | | 1 in sync, 0 not, -1 unknown |
| `pdm_model_threshold` | gauge | | Decision threshold of the served model |
| `pdm_drift_psi` | gauge | sensor | PSI of each sensor |
| `pdm_drift_max_psi` | gauge | | Highest sensor PSI (-1 while fewer than `min_rows` rows) |
| `pdm_drift_detected` | gauge | | 1 if `max_psi` > threshold |
| `pdm_drift_threshold`, `pdm_drift_window_rows` | gauge | | Alarm threshold, rows in the window |
| `pdm_alerts_received_total` | counter | alertname, status | Notifications received from Alertmanager |

### 3.11 `GET /drift/status` 🔒
PSI of the latest `window_rows` sensor readings received by `/predict` and `/predict/csv`,
against the training data of the model being served.
- **200**
  ```json
  {"state": "drift", "n_rows": 3054, "min_rows": 3000, "window_rows": 5000, "threshold": 1.0,
   "baseline_model_version": "7", "max_psi": 4.5754, "worst_sensor": "s17",
   "drift_detected": true, "sensor_psi": {"s2": 3.12, "s3": 2.87, …}}
  ```
  `state`: `no_baseline` (model trained before monitoring existed) | `collecting` (fewer than
  `min_rows` rows; `max_psi` is null) | `ok` | `drift`.

### 3.12 `POST /drift/reset` 🔒
Empties the drift window (for example after a drift alert has been handled). Returns the same body as 3.11.
The window is also emptied automatically when a different model version is loaded.

### 3.13 `POST /alerts/webhook` 🔒
Called by Alertmanager. Body: Alertmanager's webhook format (`status`, `alerts[]` with `status`,
`labels`, `annotations`). Each alert is logged (firing = WARNING, resolved = INFO), counted in
`pdm_alerts_received_total` and kept in memory (latest 50).
- **200** `{"received": 1}`

### 3.14 `GET /alerts/recent?limit=20` 🔒
Alerts received, newest first (`limit` 1 to 50).
- **200** `{"alerts": [{"received_at": "2026-11-07T13:34:02+05:30", "alertname": "DataDrift",
  "status": "firing", "severity": "warning", "summary": "Incoming sensor data has drifted …"}]}`

## 4. PredictionResponse

```json
{
  "model_version": "6",
  "threshold": 0.5308,
  "n_engines": 2,
  "engines": [
    {
      "unit": 3,
      "cycles_received": 126,
      "last_cycle": 126,
      "failure_probability": 0.92,
      "at_risk": true,
      "risk_level": "high",
      "explanation": {
        "summary": "High risk of failure within 30 cycles. Main reasons: HPC outlet static pressure (Ps30) and HPC outlet temperature (T30) are raising the risk.",
        "top_factors": [
          {"sensor": "s11", "description": "HPC outlet static pressure (Ps30)",
           "effect": "raises risk", "contribution": 1.84}
        ]
      },
      "warnings": []
    }
  ],
  "warnings": []
}
```

Rules:
- The prediction for an engine is made at its **last received cycle**, using that cycle and up to 9 earlier ones (window 10).
- `at_risk` = `failure_probability >= threshold` (threshold stored with the model version).
- `risk_level`: `high` if at risk; `medium` if probability ≥ threshold / 2; otherwise `low`.
- `explanation.top_factors`: SHAP values of the 42 features summed per sensor (raw + mean + std),
  the 3 sensors with the largest absolute total, positive = raises risk.
- `explanation.summary` wording: high = "High risk of failure within 30 cycles. Main reasons: … raising the risk.";
  medium = "Medium risk: keep this engine under watch. Main reasons: …";
  low = "Low risk: the recent sensor readings look like a healthy engine."
- Engine warning when fewer than 10 cycles are received: the prediction uses less history and is less reliable.

## 5. Errors

Every error has the body `{"error": "<code>", "detail": "<human-readable message>"}`.

| HTTP | `error` | When |
|---|---|---|
| 400 | `bad_file` | Not a `.csv`, empty, unreadable, or larger than `MAX_UPLOAD_MB` |
| 401 | `unauthorized` | `API_KEY` is set and `X-API-Key` is missing or wrong |
| 422 | `invalid_input` | Missing required columns; non-numeric values; missing values; `unit`/`cycle` not positive whole numbers; repeated `(unit, cycle)`; no rows; more than `MAX_ROWS` rows |
| 503 | `model_unavailable` | No model loaded, or the model server cannot be reached |
| 503 | `service_unavailable` | Another service an endpoint needs (the MLflow tracking server) cannot be reached |
| 500 | `internal_error` | Unexpected failure (logged with details; message kept generic) |

## 6. Internal design (classes and modules)

| Module | Contents |
|---|---|
| `src/api/settings.py` | `Settings` dataclass built from environment variables and params.yaml |
| `src/api/engine_data.py` | `read_csv_upload()`, `readings_to_frame()`, `validate_readings()`; raises `InputValidationError` |
| `src/api/model_service.py` | `ModelService` class: loads the champion from the registry (retrying automatically if missing), predicts (through the model server or its own copy), explains with SHAP, checks the model server is in sync |
| `src/api/prediction.py` | `predict_engines()`: features (shared `compute_features_pandas`) → last cycle per engine → probability → risk level → explanation |
| `src/api/pipeline_info.py` | `pipeline_status()`, `mlflow_runs()`, `recent_log_events()` for the pipeline page |
| `src/api/schemas.py` | Pydantic request and response models (this document's sections 3 and 4) |
| `src/api/metrics.py` | Prometheus metric definitions, `route_label()`, `record_predictions()`, `update_state_gauges()` |
| `src/monitoring/drift.py` | `build_baseline()`, `bin_fractions()`, `psi()`, `psi_per_sensor()`; `DriftMonitor` class (sliding window of readings, thread-safe) |
| `src/monitoring/pipeline_metrics.py` | `push_stage_metrics()`: DVC stages push run time, success and results to the Pushgateway |
| `src/api/main.py` | FastAPI app: routes, API-key check, CORS, request logging and metrics, error handlers |

Paradigm: functional modules for data handling and features (`engine_data`, `prediction`, `feature_logic`);
classes where state must be kept (`Settings`, `ModelService`, `DriftMonitor`).

The drift baseline is saved by `train.py` as `drift/baseline.json` in the model's MLflow run.
`ModelService.load()` reads it with the model, so each model version is compared with its own
training data.
