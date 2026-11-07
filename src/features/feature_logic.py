"""
Feature definitions shared by the whole project.

The model is trained on features computed by Spark (build_features.py), but
at prediction time the API receives a small CSV from one operator and must
compute the SAME features without starting Spark (Spark takes seconds to
start, an API must answer in milliseconds). This module is the pandas
version used by the API.

If the two versions ever disagreed (for example on window size or on how the
first cycle is handled), the model would receive inputs that look different
from what it was trained on. This is called training-serving skew, and it
fails silently: no error, just worse predictions. tests/test_features.py
runs both versions on the same data and checks the results match.

The rules both versions follow:
    - Features are computed per engine, in cycle order.
    - For each sensor: the raw value, the mean of the last `window` cycles,
      and the sample standard deviation of the last `window` cycles.
    - Early cycles use whatever history exists (cycle 3 averages cycles 1-3).
    - The standard deviation of a single value is undefined, so it is set to 0.
"""

import pandas as pd


def feature_names(sensors: list, window: int) -> list:
    """Names of all model input columns, in a fixed order."""
    names = list(sensors)
    names += [f"{s}_mean{window}" for s in sensors]
    names += [f"{s}_std{window}" for s in sensors]
    return names


def compute_features_pandas(df: pd.DataFrame, sensors: list, window: int) -> pd.DataFrame:
    """
    Add rolling mean and std columns for each sensor, per engine.

    Expects columns "unit", "cycle" and every sensor in `sensors`.
    Returns a new DataFrame sorted by unit and cycle.
    """
    missing = [c for c in ["unit", "cycle"] + list(sensors) if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit")[sensors]

    # rolling(...) restarts for every unit because of the groupby, so one
    # engine's history never leaks into another engine's features.
    roll = grouped.rolling(window, min_periods=1)
    rolling_mean = roll.mean().reset_index(level=0, drop=True)
    # ddof=1 (sample std) matches Spark's stddev_samp
    rolling_std = roll.std(ddof=1).reset_index(level=0, drop=True).fillna(0.0)

    for s in sensors:
        df[f"{s}_mean{window}"] = rolling_mean[s]
        df[f"{s}_std{window}"] = rolling_std[s]
    return df
