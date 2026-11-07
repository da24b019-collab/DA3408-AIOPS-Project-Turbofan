"""
Settings for the API, read from environment variables.

Everything that changes between environments (a laptop, a Docker container,
a test) is configurable here instead of being written into the code. For
example, inside Docker the model server is not at 127.0.0.1 but at a
container name, so only MODEL_SERVER_URL changes, not the code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PROJECT_ROOT, load_params
from src.features.feature_logic import feature_names


def _split_csv(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    mlflow_tracking_uri: str
    model_name: str
    model_alias: str
    model_server_url: str        # "" means: predict with the backend's own model copy
    api_key: str                 # "" means: no key required (development)
    cors_origins: list = field(default_factory=list)
    max_upload_mb: float = 5.0
    max_rows: int = 50_000
    sensors: list = field(default_factory=list)
    window: int = 10
    experiment_name: str = "turbofan-failure-prediction"
    # Where dvc.yaml, reports/ and the DVC data live. In Docker the code is in
    # the image (/app) while the project folder is mounted at /project.
    project_dir: Path = PROJECT_ROOT

    @property
    def features(self) -> list:
        return feature_names(self.sensors, self.window)

    @classmethod
    def from_env(cls) -> "Settings":
        """Defaults come from params.yaml; environment variables override them."""
        params = load_params()
        return cls(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]),
            model_name=os.getenv("MODEL_NAME", params["mlflow"]["registered_model_name"]),
            model_alias=os.getenv("MODEL_ALIAS", "champion"),
            model_server_url=os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:5001").rstrip("/"),
            api_key=os.getenv("API_KEY", ""),
            cors_origins=_split_csv(os.getenv(
                "CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")),
            max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "5")),
            max_rows=int(os.getenv("MAX_ROWS", "50000")),
            sensors=list(params["features"]["sensors"]),
            window=int(params["features"]["window"]),
            experiment_name=os.getenv("EXPERIMENT_NAME", params["mlflow"]["experiment_name"]),
            project_dir=Path(os.getenv("PROJECT_DIR", PROJECT_ROOT)),
        )
