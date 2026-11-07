"""
Tests for the dashboard's own logic (formatting, sorting, error handling).

The frontend is a separate program in frontend/, so we add that folder to
the import path, exactly as Streamlit does when it runs frontend/app.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from api_client import ApiClient, ApiError  # noqa: E402
from views.pipeline import dag_dot  # noqa: E402
from views.predict import probability_text, results_table  # noqa: E402


def engine(unit, prob, level, warnings=()):
    return {"unit": unit, "failure_probability": prob, "risk_level": level,
            "cycles_received": 20, "last_cycle": 20, "warnings": list(warnings),
            "explanation": {"summary": "", "top_factors": [
                {"sensor": "s11", "description": "HPC outlet static pressure (Ps30)",
                 "effect": "raises risk", "contribution": 1.0}]}}


def test_results_are_sorted_riskiest_first():
    table = results_table({"engines": [engine(1, 0.1, "low"), engine(2, 0.9, "high"),
                                       engine(3, 0.4, "medium")]})
    assert list(table["Engine"]) == [2, 3, 1]


def test_low_risk_engines_show_no_reason():
    table = results_table({"engines": [engine(1, 0.01, "low")]})
    assert table.loc[0, "Main reason"] == "-"


def test_warnings_appear_in_notes():
    table = results_table({"engines": [engine(1, 0.3, "medium", ["Only 4 cycle(s) received"])]})
    assert "Only 4 cycle(s)" in table.loc[0, "Notes"]


@pytest.mark.parametrize("prob, text", [(0.9999, "over 99%"), (0.001, "under 1%"), (0.42, "42%")])
def test_probability_never_claims_certainty(prob, text):
    assert probability_text(prob) == text


def test_pipeline_diagram_has_every_stage_and_arrow():
    status = {"stages": [{"name": "ingest", "status": "up_to_date"},
                         {"name": "train", "status": "changed"}],
              "edges": [["data/raw.dvc", "ingest"], ["ingest", "train"]]}
    dot = dag_dot(status)
    assert '"ingest" -> "train";' in dot
    assert "needs re-run" in dot  # the changed stage is labelled


def test_unreachable_backend_gives_friendly_error():
    client = ApiClient(base_url="http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(ApiError) as err:
        client.model_info()
    assert err.value.code == "connection_error"
    assert "Is the backend running?" in str(err.value)
