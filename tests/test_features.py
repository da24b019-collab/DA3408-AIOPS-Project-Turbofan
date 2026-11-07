"""
Unit tests for feature engineering.

The pandas tests check the feature rules on tiny examples worked out by hand.
The last test is the most important one: it runs the Spark version and the
pandas version on the same data and checks they agree. If someone changes
one version and forgets the other, this test fails (training-serving skew).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_logic import compute_features_pandas, feature_names


def make_engine(unit: int, values: list) -> pd.DataFrame:
    """One engine with a single sensor s2 taking the given values."""
    return pd.DataFrame(
        {
            "unit": unit,
            "cycle": range(1, len(values) + 1),
            "s2": [float(v) for v in values],
        }
    )


# ---------------------------------------------------------------------------
# pandas version
# ---------------------------------------------------------------------------
def test_feature_names_order():
    assert feature_names(["s2", "s3"], 5) == [
        "s2",
        "s3",
        "s2_mean5",
        "s3_mean5",
        "s2_std5",
        "s3_std5",
    ]


def test_rolling_mean_by_hand():
    df = compute_features_pandas(make_engine(1, [1, 2, 3, 4, 5]), ["s2"], window=3)
    # cycle 1: mean(1)=1, cycle 2: mean(1,2)=1.5, then (1,2,3)=2, (2,3,4)=3, (3,4,5)=4
    assert df["s2_mean3"].tolist() == [1.0, 1.5, 2.0, 3.0, 4.0]


def test_first_cycle_std_is_zero():
    # The std of a single value is undefined; both versions must use 0.
    df = compute_features_pandas(make_engine(1, [7, 9]), ["s2"], window=3)
    assert df["s2_std3"].iloc[0] == 0.0
    # Sample std of (7, 9) = sqrt(2) ~ 1.414
    assert df["s2_std3"].iloc[1] == pytest.approx(np.sqrt(2))


def test_engines_do_not_mix():
    # Engine 2 starts fresh: its first mean must be its own first value (100),
    # not an average that includes engine 1's readings.
    df = pd.concat([make_engine(1, [1, 2, 3]), make_engine(2, [100, 200])])
    out = compute_features_pandas(df, ["s2"], window=3)
    engine2 = out[out["unit"] == 2]
    assert engine2["s2_mean3"].tolist() == [100.0, 150.0]


def test_unsorted_input_is_sorted():
    df = make_engine(1, [1, 2, 3]).iloc[::-1]  # reverse the row order
    out = compute_features_pandas(df, ["s2"], window=2)
    assert out["cycle"].tolist() == [1, 2, 3]
    assert out["s2_mean2"].tolist() == [1.0, 1.5, 2.5]


def test_missing_column_raises():
    with pytest.raises(ValueError):
        compute_features_pandas(make_engine(1, [1, 2]), ["s2", "s3"], window=2)


# ---------------------------------------------------------------------------
# Spark version must match pandas
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spark():
    """Start Spark once for all tests in this file (starting it takes a few seconds)."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-features")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_spark_matches_pandas(spark):
    from src.features.build_features import add_rolling_features

    # 3 random engines of different lengths, 2 sensors
    rng = np.random.default_rng(seed=0)
    parts = []
    for unit, n_cycles in [(1, 25), (2, 8), (3, 1)]:
        parts.append(
            pd.DataFrame(
                {
                    "subset": "FD001",
                    "unit": unit,
                    "cycle": range(1, n_cycles + 1),
                    "s2": rng.normal(642, 0.5, n_cycles),
                    "s3": rng.normal(1590, 6, n_cycles),
                }
            )
        )
    data = pd.concat(parts, ignore_index=True)
    sensors, window = ["s2", "s3"], 10
    cols = feature_names(sensors, window)

    expected = compute_features_pandas(data, sensors, window)
    result = (
        add_rolling_features(spark.createDataFrame(data), sensors, window)
        .toPandas()
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    # Allow only floating-point rounding differences
    np.testing.assert_allclose(
        result[cols].to_numpy(), expected[cols].to_numpy(), rtol=1e-6, atol=1e-6
    )
