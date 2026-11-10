"""
Performance test (test case TC-P01 and TC-P02 in docs/test_cases.md).

Sends real requests to the running backend and reports how long they take:
    TC-P01  one engine as JSON, 100 times        (what an operator does most)
    TC-P02  a CSV with 10 engines, 20 times      (a batch upload, with SHAP reasons)

The backend uses the MLflow model server for the probabilities, so the times
include the whole path: API -> validation -> features -> model server -> SHAP.

Run (with `docker compose up -d` running):
    python -m scripts.latency_test
"""

import json
import statistics
import time
import urllib.request
import uuid

import pandas as pd

from src.config import PROJECT_ROOT
from src.retraining.worker import read_api_key

BACKEND_URL = "http://127.0.0.1:8000"
SAMPLE = PROJECT_ROOT / "samples" / "fd001_mixed_engines.csv"

# Acceptance criteria (docs/test_plan.md, section 7)
TARGET_SINGLE_P95_MS = 200
TARGET_BATCH_P95_MS = 500


def post(path: str, body: bytes, content_type: str) -> float:
    """Send one request; return the time it took in milliseconds."""
    request = urllib.request.Request(
        BACKEND_URL + path,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "X-API-Key": read_api_key()},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
        assert response.status == 200
    return (time.perf_counter() - start) * 1000


def csv_upload_body(csv_bytes: bytes) -> tuple:
    """A multipart/form-data body with one file field, built by hand."""
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="engines.csv"\r\nContent-Type: text/csv\r\n\r\n'
    ).encode()
    body += csv_bytes + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def summary(name: str, times: list, target_ms: float) -> bool:
    times = sorted(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    passed = p95 < target_ms
    print(
        f"{name:34s} n={len(times):3d}  median {statistics.median(times):7.1f} ms  "
        f"p95 {p95:7.1f} ms  max {times[-1]:7.1f} ms  "
        f"target p95 < {target_ms} ms -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def main() -> None:
    data = pd.read_csv(SAMPLE)
    one_engine = data[data["unit"] == data["unit"].iloc[0]]
    ten_engines = data[data["unit"].isin(data["unit"].unique()[:10])]
    json_body = json.dumps({"readings": one_engine.to_dict(orient="records")}).encode()
    csv_body, csv_type = csv_upload_body(ten_engines.to_csv(index=False).encode())

    post("/predict", json_body, "application/json")  # warm-up (first call loads caches)

    single = [post("/predict", json_body, "application/json") for _ in range(100)]
    batch = [post("/predict/csv", csv_body, csv_type) for _ in range(20)]

    print(
        f"Engine rows: single request {len(one_engine)}, batch {len(ten_engines)} "
        f"({ten_engines['unit'].nunique()} engines)\n"
    )
    ok = summary("TC-P01 one engine (JSON)", single, TARGET_SINGLE_P95_MS)
    ok &= summary("TC-P02 10 engines (CSV upload)", batch, TARGET_BATCH_P95_MS)
    print("\nAll performance criteria met." if ok else "\nA performance criterion was NOT met.")


if __name__ == "__main__":
    main()
