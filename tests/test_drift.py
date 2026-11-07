"""
Tests for the PSI drift detector (src/monitoring/drift.py).

Synthetic data with a known answer is used, so each test checks one
property of PSI that we rely on when explaining the alarm.
"""

import numpy as np
import pandas as pd

from src.monitoring.drift import DriftMonitor, bin_fractions, build_baseline, psi

SENSORS = ["s2", "s3"]


def normal_data(n: int, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({s: rng.normal(100 + shift, 1, n) for s in SENSORS})


def test_baseline_bins_hold_ten_percent_each():
    baseline = build_baseline(normal_data(10_000), SENSORS)
    expected = baseline["s2"]["expected"]
    assert len(expected) == 10
    assert np.allclose(expected, 0.1, atol=0.002)


def test_repeated_values_do_not_create_duplicate_edges():
    # A sensor with only 3 distinct values (like s17) cannot have 9 distinct deciles
    data = pd.DataFrame({"s2": [1.0] * 50 + [2.0] * 30 + [3.0] * 20, "s3": range(100)})
    baseline = build_baseline(data, SENSORS)
    edges = baseline["s2"]["edges"]
    assert edges == sorted(set(edges))
    assert abs(sum(baseline["s2"]["expected"]) - 1) < 1e-9


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == 0


def test_psi_grows_with_the_shift():
    baseline = build_baseline(normal_data(10_000), SENSORS)
    edges, expected = baseline["s2"]["edges"], baseline["s2"]["expected"]
    small = psi(expected, bin_fractions(normal_data(5000, 0.2, 1)["s2"].to_numpy(), edges))
    large = psi(expected, bin_fractions(normal_data(5000, 2.0, 1)["s2"].to_numpy(), edges))
    assert 0 < small < large
    assert large > 1.0


def test_empty_bins_do_not_give_infinity():
    # All new readings far above the training range: 9 of 10 bins are empty
    value = psi([0.1] * 10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
    assert np.isfinite(value) and value > 1


def test_monitor_states():
    monitor = DriftMonitor(SENSORS, threshold=1.0, min_rows=100, window_rows=200)
    assert monitor.status()["state"] == "no_baseline"

    monitor.set_baseline(build_baseline(normal_data(10_000), SENSORS), "7")
    monitor.add(normal_data(50, seed=1))
    assert monitor.status()["state"] == "collecting"  # fewer than min_rows

    monitor.add(normal_data(150, seed=2))
    status = monitor.status()
    assert status["state"] == "ok" and not status["drift_detected"]

    monitor.add(normal_data(200, shift=3.0, seed=3))  # replaces the whole window
    status = monitor.status()
    assert status["state"] == "drift" and status["drift_detected"]
    assert status["n_rows"] == 200  # window never grows past its size
    assert status["max_psi"] > 1.0


def test_new_baseline_empties_the_window():
    monitor = DriftMonitor(SENSORS, min_rows=10, window_rows=100)
    monitor.set_baseline(build_baseline(normal_data(1000), SENSORS), "7")
    monitor.add(normal_data(50))
    monitor.set_baseline(build_baseline(normal_data(1000), SENSORS), "8")
    assert monitor.status()["n_rows"] == 0
    assert monitor.status()["baseline_model_version"] == "8"


def test_closest_of_several_baselines_is_used():
    # Model trained on two datasets: one around 100, one around 110
    baselines = {
        "A": build_baseline(normal_data(5000, 0.0, seed=1), SENSORS),
        "B": build_baseline(normal_data(5000, 10.0, seed=2), SENSORS),
    }
    monitor = DriftMonitor(SENSORS, threshold=1.0, min_rows=100, window_rows=500)
    monitor.set_baseline(baselines, "8")
    monitor.add(normal_data(500, 10.0, seed=3))  # looks like B only
    status = monitor.status()
    assert status["state"] == "ok" and status["closest_baseline"] == "B"
    assert status["baselines"] == ["A", "B"]


def test_old_single_baseline_format_still_works():
    monitor = DriftMonitor(SENSORS, min_rows=10, window_rows=100)
    monitor.set_baseline(build_baseline(normal_data(1000), SENSORS), "7")  # pre-Task 9 format
    assert monitor.status()["baselines"] == ["training"]
