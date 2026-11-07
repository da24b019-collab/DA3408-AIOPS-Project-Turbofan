"""
Experiment: which PSI threshold and window size should the drift alarm use?

The textbook rule says PSI > 0.25 means "significant shift". This script
checks whether that rule fits OUR data, by measuring the max PSI (over the
14 sensors) for windows of readings that are known to be:
    normal  -> FD001 test engines (same conditions and fault as training)
    drift   -> FD003 (same conditions, plus a second fault mode)
               FD002 (six operating conditions instead of one)

Each window is a random set of WHOLE engines cut to the window size,
because that is how data reaches the API: one engine's history per upload.
(Sampling random single rows from all engines gives much lower PSI for
small windows, and would make a small window look safe when it is not.)

For each window size it prints how often the max PSI goes above the
threshold: for normal data that is the false-alarm rate, for FD002/FD003
it is the detection rate.

Run:  python -m experiments.drift_threshold_study
"""

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_params
from src.monitoring.drift import build_baseline, psi_per_sensor

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPEATS = 100
WINDOW_SIZES = [50, 500, 1000, 3000, 5000]


def random_window(engines: list, n_rows: int, rng) -> pd.DataFrame:
    """Whole engines in random order, then the first n_rows rows of them."""
    order = rng.permutation(len(engines))
    return pd.concat([engines[i] for i in order]).head(n_rows)


def main():
    params = load_params()
    sensors = params["features"]["sensors"]
    threshold = params["monitoring"]["drift_threshold"]
    rng = np.random.default_rng(0)

    train = pd.read_parquet(FEATURES_DIR / "train_FD001.parquet")
    baseline = build_baseline(train, sensors, params["monitoring"]["psi_bins"])

    cases = {
        "FD001 test (normal)": "test_FD001.parquet",
        "FD003 test (new fault)": "test_FD003.parquet",
        "FD002 test (new conditions)": "test_FD002.parquet",
    }
    # Split each file into a list of engines once, so windows are quick to build
    engines = {name: [g for _, g in pd.read_parquet(FEATURES_DIR / f).groupby("unit")]
               for name, f in cases.items()}

    print(f"Max PSI over {len(sensors)} sensors, {REPEATS} random windows per case, "
          f"threshold {threshold}\n")
    print(f"{'case':30s} {'rows':>5s} {'min':>6s} {'median':>6s} {'p95':>6s} {'max':>6s} "
          f"{'above threshold':>16s}")
    for n_rows in WINDOW_SIZES:
        for name in cases:
            values = np.array([max(psi_per_sensor(baseline, random_window(engines[name],
                                                                            n_rows, rng)).values())
                               for _ in range(REPEATS)])
            print(f"{name:30s} {n_rows:5d} {values.min():6.2f} {np.median(values):6.2f} "
                  f"{np.percentile(values, 95):6.2f} {values.max():6.2f} "
                  f"{(values > threshold).mean():15.0%}")
        print()


if __name__ == "__main__":
    main()
