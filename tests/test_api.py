"""
Tests for the FastAPI backend (docs/lld_api.md).

A FakeModelService replaces the real one, so these tests need no MLflow
server and no trained model: they check the API's own behaviour
(validation, status codes, response shape, API key, risk levels).
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.engine_data import InputValidationError, validate_readings
from src.api.main import create_app
from src.api.model_service import ModelUnavailableError
from src.api.prediction import risk_level
from src.api.settings import Settings

SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
           "s13", "s14", "s15", "s17", "s20", "s21"]


def make_settings(api_key: str = "") -> Settings:
    return Settings(mlflow_tracking_uri="http://unused", model_name="test-model",
                    model_alias="champion", model_server_url="", api_key=api_key,
                    cors_origins=[], sensors=SENSORS, window=10)


class FakeModelService:
    """Predictable stand-in: risk grows with the recent average of sensor s11."""

    def __init__(self, settings, loaded=True, server_status=("not_configured", None)):
        self.settings = settings
        self._loaded = loaded
        self._server_status = server_status
        self.info = {"version": "7", "threshold": 0.5} if loaded else {}

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        raise ModelUnavailableError("registry unreachable")

    def load_if_missing(self):
        pass

    def predict_proba(self, X):
        if not self._loaded:
            raise ModelUnavailableError("No model is loaded.")
        # s11 around 47 = healthy, around 48 = worn: map to a 0..1 probability
        return ((X["s11_mean10"] - 47.0).clip(0, 1)).to_numpy()

    def explain(self, row):
        return [{"sensor": "s11", "description": "HPC outlet static pressure (Ps30)",
                 "effect": "raises risk", "contribution": 1.0}]

    def model_server_status(self):
        return self._server_status


def engine_rows(unit: int, n_cycles: int, s11: float) -> list:
    """Readings for one engine with every sensor at a constant value."""
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = {"unit": unit, "cycle": cycle, **{s: 100.0 for s in SENSORS}}
        row["s11"] = s11
        rows.append(row)
    return rows


@pytest.fixture
def client():
    settings = make_settings()
    with TestClient(create_app(settings, FakeModelService(settings))) as c:
        yield c


# ------------------------------------------------------------------ status
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_not_ready_without_model():
    settings = make_settings()
    with TestClient(create_app(settings, FakeModelService(settings, loaded=False))) as c:
        response = c.get("/ready")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False
        # Predicting without a model gives a clear 503, not a crash
        assert c.post("/predict", json={"readings": engine_rows(1, 3, 47.0)}).status_code == 503


def test_not_ready_when_model_server_out_of_sync():
    settings = make_settings()
    fake = FakeModelService(settings, server_status=("reachable", False))
    with TestClient(create_app(settings, fake)) as c:
        assert c.get("/ready").status_code == 503


# -------------------------------------------------------------- prediction
def test_predict_json_two_engines(client):
    readings = engine_rows(1, 12, s11=47.0) + engine_rows(2, 12, s11=48.0)
    body = client.post("/predict", json={"readings": readings}).json()
    assert body["n_engines"] == 2
    healthy, worn = body["engines"]
    assert healthy["risk_level"] == "low" and healthy["at_risk"] is False
    assert worn["risk_level"] == "high" and worn["at_risk"] is True
    assert worn["explanation"]["top_factors"][0]["sensor"] == "s11"
    assert worn["last_cycle"] == 12 and worn["warnings"] == []


def test_predict_csv_upload(client):
    csv = pd.DataFrame(engine_rows(5, 10, s11=48.0)).to_csv(index=False).encode()
    response = client.post("/predict/csv", files={"file": ("engines.csv", io.BytesIO(csv))})
    assert response.status_code == 200
    assert response.json()["engines"][0]["unit"] == 5


def test_short_history_gives_warning(client):
    body = client.post("/predict", json={"readings": engine_rows(1, 4, 47.0)}).json()
    assert "Only 4 cycle(s)" in body["engines"][0]["warnings"][0]


# -------------------------------------------------------------- bad input
def test_missing_column_is_422(client):
    rows = engine_rows(1, 3, 47.0)
    for row in rows:
        del row["s11"]
    response = client.post("/predict", json={"readings": rows})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_input",
                               "detail": "Missing required column(s): s11."}


def test_wrong_json_shape_uses_same_error_format(client):
    response = client.post("/predict", json={"wrong": []})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_input"


def test_non_csv_upload_is_400(client):
    response = client.post("/predict/csv", files={"file": ("notes.txt", io.BytesIO(b"hello"))})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_file"


def test_duplicate_cycle_is_422(client):
    rows = engine_rows(1, 3, 47.0) + engine_rows(1, 1, 47.0)  # cycle 1 twice
    response = client.post("/predict", json={"readings": rows})
    assert response.status_code == 422
    assert "more than once" in response.json()["detail"]


# ----------------------------------------------------------------- API key
def test_api_key_required_when_set():
    settings = make_settings(api_key="secret")
    with TestClient(create_app(settings, FakeModelService(settings))) as c:
        body = {"readings": engine_rows(1, 3, 47.0)}
        assert c.post("/predict", json=body).status_code == 401
        assert c.post("/predict", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/predict", json=body, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200  # monitoring stays open


# ------------------------------------------------------------ pure functions
def test_risk_level_boundaries():
    assert risk_level(0.50, threshold=0.5) == "high"     # equal to threshold counts as at risk
    assert risk_level(0.25, threshold=0.5) == "medium"   # half the threshold
    assert risk_level(0.24, threshold=0.5) == "low"


def test_column_names_are_case_insensitive():
    df = pd.DataFrame(engine_rows(1, 2, 47.0)).rename(columns={"s11": " S11 "})
    clean = validate_readings(df, SENSORS, max_rows=100)
    assert "s11" in clean.columns


def test_too_many_rows():
    df = pd.DataFrame(engine_rows(1, 5, 47.0))
    with pytest.raises(InputValidationError):
        validate_readings(df, SENSORS, max_rows=4)


def test_summary_lists_reasons_naturally():
    from src.api.prediction import summary_sentence
    factors = [{"description": d, "effect": "raises risk"} for d in ("A", "B", "C")]
    assert summary_sentence("high", factors).endswith("Main reasons: A, B and C are raising the risk.")
    assert summary_sentence("high", factors[:1]).endswith("Main reasons: A is raising the risk.")


# ------------------------------------------------------------ monitoring (Task 8)
from src.monitoring.drift import build_baseline  # noqa: E402


def drift_client(api_key: str = ""):
    """App whose fake model has a drift baseline: every sensor near 100 in training."""
    settings = make_settings(api_key)
    settings.drift_min_rows, settings.drift_window_rows = 300, 600
    fake = FakeModelService(settings)
    rng = __import__("numpy").random.default_rng(0)
    training = pd.DataFrame({s: rng.normal(100, 1, 2000) for s in SENSORS})
    fake.drift_baseline = build_baseline(training, SENSORS)
    return TestClient(create_app(settings, fake))


def noisy_engine(unit: int, n_cycles: int, centre: float) -> list:
    rng = __import__("numpy").random.default_rng(unit)
    rows = []
    for cycle in range(1, n_cycles + 1):
        rows.append({"unit": unit, "cycle": cycle,
                     **{s: float(rng.normal(centre, 1)) for s in SENSORS}})
    return rows


def test_metrics_page_is_open_and_counts_predictions(client):
    client.post("/predict", json={"readings": engine_rows(1, 12, 47.9)})
    page = client.get("/metrics")
    assert page.status_code == 200
    assert "text/plain" in page.headers["content-type"]
    text = page.text
    assert 'pdm_http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "pdm_predictions_total" in text
    assert "pdm_model_loaded 1.0" in text


def test_unknown_paths_are_grouped_in_metrics(client):
    client.get("/no-such-page-123")
    assert 'path="other"' in client.get("/metrics").text
    assert "no-such-page-123" not in client.get("/metrics").text


def test_validation_errors_are_counted(client):
    client.post("/predict", json={"readings": [{"unit": 1}]})
    assert 'pdm_validation_errors_total{error="invalid_input"}' in client.get("/metrics").text


def test_drift_without_baseline(client):
    assert client.get("/drift/status").json()["state"] == "no_baseline"


def test_drift_collecting_then_ok_then_drift():
    with drift_client() as c:
        c.post("/predict", json={"readings": noisy_engine(1, 100, 100)})
        assert c.get("/drift/status").json()["state"] == "collecting"   # 100 < 300 rows

        # Same distribution as training. (With only ~40 rows this would wrongly
        # show "drift": small samples inflate PSI, the reason for min_rows.)
        c.post("/predict", json={"readings": noisy_engine(2, 400, 100)})
        assert c.get("/drift/status").json()["state"] == "ok"

        c.post("/predict", json={"readings": noisy_engine(3, 600, 105)})  # far from training
        status = c.get("/drift/status").json()
        assert status["state"] == "drift" and status["drift_detected"]
        assert "pdm_drift_detected 1.0" in c.get("/metrics").text

        assert c.post("/drift/reset").json()["n_rows"] == 0


def test_alert_webhook_accepts_bearer_token_and_stores_alert():
    alert = {"status": "firing", "alerts": [{
        "status": "firing", "labels": {"alertname": "DataDrift", "severity": "warning"},
        "annotations": {"summary": "Sensor data has drifted"}}]}
    with drift_client(api_key="secret") as c:
        assert c.post("/alerts/webhook", json=alert).status_code == 401
        ok = c.post("/alerts/webhook", json=alert, headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200 and ok.json() == {"received": 1}
        recent = c.get("/alerts/recent", headers={"X-API-Key": "secret"}).json()["alerts"]
        assert recent[0]["alertname"] == "DataDrift" and recent[0]["status"] == "firing"
        # /metrics stays open even when an API key is set
        assert c.get("/metrics").status_code == 200
