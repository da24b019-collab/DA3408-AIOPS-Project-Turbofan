# Low-Level Design: Backend API

## 1. Components

| Component | Port | Role |
|---|---|---|
| MLflow model server (`mlflow models serve`) | 5001 | Inference engine. Turns 42 feature values into `[P(safe), P(at risk)]`. Serves the registry version with alias `champion`. |
| FastAPI backend (`src/api`) | 8000 | Business layer used by the frontend. Validates input, computes features, calls the model server, applies the threshold, explains predictions, reports model information. |
| MLflow tracking server | 5000 | Model registry. The backend reads the champion's version, threshold and feature list from it. |

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

## 3. Endpoints

All responses are JSON. Protected endpoints (marked 🔒) need `X-API-Key` when `API_KEY` is set.

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
  `model_server`: `reachable` | `unreachable` | `not_configured`.
  `model_server_in_sync`: the model server and the backend's copy give identical
  probabilities on a fixed check input (`null` if the server is not reachable).
- **503** same body with `"ready": false` when no model is loaded, when the model server
  is configured but unreachable (predictions would fail), or when it is out of sync
  (its predictions would not match the threshold and explanations of the loaded version).
  After approving a new champion: restart the model server, then call `/model/reload`.

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
| 500 | `internal_error` | Unexpected failure (logged with details; message kept generic) |

## 6. Internal design (classes and modules)

| Module | Contents |
|---|---|
| `src/api/settings.py` | `Settings` dataclass built from environment variables and params.yaml |
| `src/api/engine_data.py` | `read_csv_upload()`, `readings_to_frame()`, `validate_readings()`; raises `InputValidationError` |
| `src/api/model_service.py` | `ModelService` class: loads the champion from the registry, predicts (through the model server or its own copy), explains with SHAP, checks the model server is in sync |
| `src/api/prediction.py` | `predict_engines()`: features (shared `compute_features_pandas`) → last cycle per engine → probability → risk level → explanation |
| `src/api/schemas.py` | Pydantic request and response models (this document's sections 3 and 4) |
| `src/api/main.py` | FastAPI app: routes, API-key check, CORS, request logging, error handlers |

Paradigm: functional modules for data handling and features (`engine_data`, `prediction`, `feature_logic`);
classes where state must be kept (`Settings`, `ModelService`).
