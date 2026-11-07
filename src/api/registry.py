"""
The model registry operations the API needs for approval and rollback.

A thin class around MlflowClient, so the tests can replace it with a fake
(no MLflow server needed to test the approval rules).
"""

import threading

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.compare import compare_versions


class RegistryError(Exception):
    """The registry cannot be reached."""


class VersionNotFound(Exception):
    """No such model version."""


class Registry:
    def __init__(self, settings, max_fd001_drop: float):
        self.settings = settings
        self.max_fd001_drop = max_fd001_drop
        self._cache = {}  # (champion, challenger) -> comparison result
        self._lock = threading.Lock()  # one comparison at a time (it loads two models)

    def _client(self) -> MlflowClient:
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        return MlflowClient()

    def alias_version(self, alias: str):
        """Version number holding the alias, or None."""
        try:
            return str(
                self._client().get_model_version_by_alias(self.settings.model_name, alias).version
            )
        except MlflowException:
            return None

    def versions(self) -> list:
        """All versions, newest first, with their aliases and key tags."""
        try:
            found = self._client().search_model_versions(f"name='{self.settings.model_name}'")
        except MlflowException as err:
            raise RegistryError(str(err)) from err
        rows = [
            {
                "version": str(v.version),
                "aliases": list(v.aliases),
                "train_subsets": v.tags.get("train_subsets", "FD001"),
                "test_recall": v.tags.get("test_recall"),
                "created": v.creation_timestamp,
            }
            for v in found
        ]
        return sorted(rows, key=lambda r: int(r["version"]), reverse=True)

    def compare(self) -> dict:
        """Champion vs challenger on all test sets (cached: loading models takes seconds)."""
        champion, challenger = self.alias_version("champion"), self.alias_version("challenger")
        if champion is None or challenger is None:
            raise RegistryError("Both a champion and a challenger are needed to compare.")
        with self._lock:
            key = (champion, challenger)
            if key not in self._cache:
                self._cache[key] = compare_versions(
                    self.settings.model_name,
                    champion,
                    challenger,
                    self.max_fd001_drop,
                    self.settings.project_dir / "data" / "features",
                )
            return self._cache[key]

    def set_champion(self, version: str) -> None:
        client = self._client()
        try:
            client.get_model_version(self.settings.model_name, version)  # must exist
            client.set_registered_model_alias(self.settings.model_name, "champion", version)
        except MlflowException as err:
            if err.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise VersionNotFound(f"Model version {version} does not exist") from err
            raise RegistryError(f"Cannot make version {version} champion: {err}") from err
