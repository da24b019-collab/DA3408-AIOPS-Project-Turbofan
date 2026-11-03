"""
Unit tests for the ingestion stage.

These use tiny hand-made tables instead of the real files, so each test
checks one rule, runs in milliseconds, and the expected answer can be
worked out by hand. Run all tests with:  pytest -v
"""

import pandas as pd
import pytest

from src.config import RAW_COLUMNS
from src.data.ingest import (
    DataValidationError,
    add_test_labels,
    add_train_labels,
    read_cmapss_file,
    validate_frame,
)


def make_engine(unit: int, n_cycles: int) -> pd.DataFrame:
    """Build a fake engine history: cycles 1..n_cycles, all sensor values 1.0."""
    rows = []
    for cycle in range(1, n_cycles + 1):
        rows.append([unit, cycle] + [1.0] * (len(RAW_COLUMNS) - 2))
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_train_rul_counts_down_to_zero():
    df = add_train_labels(make_engine(unit=1, n_cycles=5), horizon=2)
    # 5 cycles -> RUL should be 4, 3, 2, 1, 0
    assert df["rul"].tolist() == [4, 3, 2, 1, 0]


def test_label_boundary_is_inclusive():
    # RUL equal to the horizon must count as "at risk" (label 1).
    df = add_train_labels(make_engine(unit=1, n_cycles=5), horizon=2)
    assert df["label"].tolist() == [0, 0, 1, 1, 1]


def test_test_rul_adds_true_rul_at_end():
    df = make_engine(unit=1, n_cycles=3)
    true_rul = pd.Series([10], index=[1])  # engine 1 has 10 cycles left at its last row
    df = add_test_labels(df, true_rul, horizon=11)
    # last row: 10, middle: 11, first: 12
    assert df["rul"].tolist() == [12, 11, 10]
    assert df["label"].tolist() == [0, 1, 1]


def test_test_labels_reject_wrong_rul_length():
    df = pd.concat([make_engine(1, 3), make_engine(2, 3)])
    with pytest.raises(DataValidationError):
        add_test_labels(df, pd.Series([5], index=[1]), horizon=30)


def test_validation_catches_missing_cycle():
    df = make_engine(unit=1, n_cycles=5)
    df = df[df["cycle"] != 3]  # drop cycle 3 to create a gap
    with pytest.raises(DataValidationError):
        validate_frame(df, "broken")


def test_validation_catches_missing_values():
    df = make_engine(unit=1, n_cycles=3)
    df.loc[0, "s2"] = None
    with pytest.raises(DataValidationError):
        validate_frame(df, "broken")


def test_reader_handles_trailing_spaces(tmp_path):
    # Real C-MAPSS lines end with two spaces. The reader must still find 26 columns.
    line = " ".join(["1", "1"] + ["0.5"] * 24) + "  \n"
    path = tmp_path / "train_fake.txt"
    path.write_text(line * 3)
    df = read_cmapss_file(path)
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 3


def test_reader_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n")
    with pytest.raises(DataValidationError):
        read_cmapss_file(path)
