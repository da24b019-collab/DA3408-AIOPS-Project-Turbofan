"""
pytest runs this file before any test. It sends all log output from the
tests to a temporary folder, so deliberate test errors (e.g. "MLflow is
down") never end up in the project's real logs or on the dashboard.
"""

import os
import tempfile

os.environ["LOG_DIR"] = tempfile.mkdtemp(prefix="turbofan-test-logs-")
