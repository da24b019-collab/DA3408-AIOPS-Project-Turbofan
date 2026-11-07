"""The worker's params.yaml edit must change only the train section."""

import pytest

from src.retraining.worker import JobFailed, set_train_subsets

PARAMS = """data:
  subsets: [FD001, FD002, FD003, FD004]   # all datasets are ingested

train:
  # comment kept
  subsets: [FD001]
  model_type: xgboost
"""


def test_only_train_subsets_change():
    new = set_train_subsets(PARAMS, ["FD001", "FD002"])
    assert "  subsets: [FD001, FD002]\n  model_type" in new
    assert "  subsets: [FD001, FD002, FD003, FD004]   # all datasets" in new  # data untouched
    assert "# comment kept" in new


def test_missing_line_fails_clearly():
    with pytest.raises(JobFailed):
        set_train_subsets("train:\n  model_type: xgboost\n", ["FD001"])
