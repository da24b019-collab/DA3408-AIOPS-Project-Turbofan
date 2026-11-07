"""
The ONLY place where the dashboard talks to the backend.

The frontend and the backend are separate programs connected only through
REST calls. The backend's address comes from the BACKEND_URL environment
variable, so the same dashboard works on a laptop (127.0.0.1:8000) and in
Docker (http://backend:8000) without code changes.

Every method returns the parsed JSON, or raises ApiError with a message
that can be shown to a non-technical user.
"""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT_SECONDS = 60


class ApiError(Exception):
    """A call to the backend failed. `code` is the backend's error code, if any."""

    def __init__(self, message: str, code: str = "", status: int = 0):
        super().__init__(message)
        self.code = code
        self.status = status


class ApiClient:
    def __init__(self, base_url: str = BACKEND_URL, api_key: str = API_KEY):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, headers=self.headers,
                                        timeout=TIMEOUT_SECONDS, **kwargs)
        except requests.ConnectionError as err:
            raise ApiError(f"Cannot reach the prediction service at {self.base_url}. "
                           "Is the backend running?", code="connection_error") from err
        except requests.Timeout as err:
            raise ApiError("The prediction service took too long to answer. "
                           "Please try again.", code="timeout") from err

        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.ok:
            return body
        # Our backend always answers errors as {"error": ..., "detail": ...}
        raise ApiError(body.get("detail", f"Unexpected answer (HTTP {response.status_code})."),
                       code=body.get("error", ""), status=response.status_code)

    # ---- status and model ----
    def ready(self) -> dict:
        # /ready answers 503 with a useful body when not ready, so read it either way
        try:
            response = requests.get(f"{self.base_url}/ready", headers=self.headers, timeout=15)
            return response.json()
        except (requests.RequestException, ValueError) as err:
            raise ApiError(f"Cannot reach the prediction service at {self.base_url}.",
                           code="connection_error") from err

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    # ---- prediction ----
    def predict_csv(self, filename: str, content: bytes) -> dict:
        files = {"file": (filename, content, "text/csv")}
        return self._request("POST", "/predict/csv", files=files)

    # ---- pipeline ----
    def pipeline_status(self) -> dict:
        return self._request("GET", "/pipeline/status")

    def pipeline_runs(self, limit: int = 30) -> dict:
        return self._request("GET", "/pipeline/runs", params={"limit": limit})

    def recent_logs(self, limit: int = 50) -> dict:
        return self._request("GET", "/logs/recent", params={"limit": limit})
