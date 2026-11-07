"""
ModelService: everything the API needs to know about the current model.

It is a class (not plain functions) because it keeps STATE between requests:
the loaded model, its version, threshold and SHAP explainer. Loading from the
registry takes a few seconds, so we load once and reuse it for every request.

Predictions go to the MLflow model server (the inference engine) when
MODEL_SERVER_URL is set. The service also keeps its own copy of the model,
because SHAP needs the real model object to explain a prediction.
"""

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import SENSOR_DESCRIPTIONS
from src.logger import get_logger

logger = get_logger(__name__, log_file="api.log")


class ModelUnavailableError(Exception):
    """No model is loaded, or the model server cannot be reached."""


class ModelService:
    def __init__(self, settings):
        self.settings = settings
        self.model = None          # the scikit-learn / XGBoost model object
        self.explainer = None      # SHAP explainer for that model
        self.info = {}             # version, threshold, features, ...
        self._lock = threading.Lock()  # a reload must not happen halfway through a request

    # ------------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> dict:
        """
        Load the version that currently holds the alias (e.g. "champion").
        If loading fails, the previously loaded model stays in use.
        """
        s = self.settings
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        uri = f"models:/{s.model_name}@{s.model_alias}"
        try:
            version = MlflowClient().get_model_version_by_alias(s.model_name, s.model_alias)
            model = mlflow.sklearn.load_model(uri)
            metadata = mlflow.models.get_model_info(uri).metadata or {}
        except (MlflowException, OSError) as err:
            raise ModelUnavailableError(f"Could not load {uri}: {err}") from err

        features = metadata.get("features", s.features)
        if list(features) != list(s.features):
            raise ModelUnavailableError(
                "The model was trained on different features than this API computes. "
                "Retrain the model or update params.yaml.")

        info = {
            "model_name": s.model_name,
            "alias": s.model_alias,
            "version": str(version.version),
            "run_id": version.run_id,
            "model_type": metadata.get("model_type", version.tags.get("model_type", "unknown")),
            "threshold": float(metadata["threshold"]),
            "window": int(metadata.get("window", s.window)),
            "n_features": len(features),
            "git_commit": version.tags.get("git_commit", "unknown"),
            "test_metrics": {
                "recall": float(version.tags.get("test_recall", "nan")),
                "pr_auc": float(version.tags.get("test_pr_auc", "nan")),
            },
            "sensors": [{"code": c, "description": SENSOR_DESCRIPTIONS[c]} for c in s.sensors],
            "loaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        explainer = self._make_explainer(model, info["model_type"])

        with self._lock:  # swap everything at once
            self.model, self.explainer, self.info = model, explainer, info
        logger.info("Loaded %s version %s (%s, threshold %.3f)",
                    s.model_name, info["version"], info["model_type"], info["threshold"])
        return info

    @staticmethod
    def _make_explainer(model, model_type: str):
        """TreeExplainer gives exact, fast explanations for tree models (XGBoost, random forest)."""
        if model_type in ("xgboost", "random_forest"):
            return shap.TreeExplainer(model)
        logger.warning("No SHAP explainer for model type '%s'; explanations disabled", model_type)
        return None

    # --------------------------------------------------------------- prediction
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of 'at risk' for each row of 42 features."""
        if not self.is_loaded:
            raise ModelUnavailableError("No model is loaded.")
        if self.settings.model_server_url:
            return self._predict_with_server(X)
        with self._lock:
            return self.model.predict_proba(X)[:, 1]

    def _predict_with_server(self, X: pd.DataFrame) -> np.ndarray:
        """Call the MLflow model server's /invocations endpoint."""
        body = json.dumps({"dataframe_split": X.to_dict(orient="split")}).encode()
        request = urllib.request.Request(
            f"{self.settings.model_server_url}/invocations", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                predictions = json.loads(response.read())["predictions"]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as err:
            raise ModelUnavailableError(f"Model server error: {err}") from err
        return np.asarray(predictions, dtype=float)[:, 1]  # column 1 = P(at risk)

    def explain(self, row: pd.DataFrame) -> list:
        """
        SHAP contribution of each sensor for ONE row (the engine's last cycle).
        A sensor has 3 features (raw, mean, std); we add them up so the user
        sees "HPC outlet temperature", not "s3_mean10".
        Returns sensors sorted by size of effect, largest first.
        """
        if self.explainer is None:
            return []
        with self._lock:
            values = self.explainer.shap_values(row)
        values = np.asarray(values)
        if values.ndim == 3:        # (rows, features, classes) for some models
            values = values[:, :, 1]
        per_feature = dict(zip(row.columns, values[0]))

        per_sensor = {}
        for feature, value in per_feature.items():
            sensor = feature.split("_")[0]   # "s3_mean10" -> "s3"
            per_sensor[sensor] = per_sensor.get(sensor, 0.0) + float(value)
        ranked = sorted(per_sensor.items(), key=lambda item: abs(item[1]), reverse=True)
        return [{"sensor": s, "description": SENSOR_DESCRIPTIONS[s],
                 "effect": "raises risk" if v > 0 else "lowers risk",
                 "contribution": round(v, 4)} for s, v in ranked]

    # ------------------------------------------------------------------ checks
    def model_server_status(self) -> tuple:
        """
        ("reachable" | "unreachable" | "not_configured", in_sync or None).
        In sync = the server and our copy give the same probabilities on a fixed
        input. If someone approves a new champion and reloads the API but forgets
        to restart the model server, this becomes False.
        """
        if not self.settings.model_server_url:
            return "not_configured", None
        if not self.is_loaded:
            return "unknown", None
        check_input = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, len(self.settings.features))),
            columns=self.settings.features)
        try:
            server_prob = self._predict_with_server(check_input)
        except ModelUnavailableError:
            return "unreachable", None
        with self._lock:
            local_prob = self.model.predict_proba(check_input)[:, 1]
        return "reachable", bool(np.allclose(server_prob, local_prob, atol=1e-6))
