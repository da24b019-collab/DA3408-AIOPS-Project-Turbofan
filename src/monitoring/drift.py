"""
Data drift detection with the Population Stability Index (PSI).

The idea in one sentence: if the sensor readings arriving at the API are
spread out differently from the readings the model was trained on, the
model is being asked about engines it has never seen, so its predictions
can no longer be trusted.

How PSI works, for one sensor:
    1. Baseline (done once, at training time): cut the training readings
       into 10 bins so that each bin holds 10% of them (the "deciles").
    2. At serving time: count what fraction of the recent readings falls
       into each of those same 10 bins.
    3. Compare the two lists of fractions:
           PSI = sum over bins of (actual - expected) * ln(actual / expected)
       Same spread -> every term is about 0 -> PSI near 0.
       Readings piled into a few bins -> large PSI.

We compute PSI for each of the 14 sensors and raise a drift flag when the
HIGHEST one goes above the threshold. Threshold 1.0, minimum 500 rows and
a 1,000-row window were chosen from measurements, not from the textbook
0.25 (see docs/monitoring_findings.md and experiments/drift_threshold_study.py).
"""

import threading
from collections import deque

import numpy as np
import pandas as pd

# A bin with no readings would give ln(0). We replace 0 with this small
# fraction so the formula stays finite (a common, simple fix).
EPS = 1e-4


# ---------------------------------------------------------------- baseline
def build_baseline(train: pd.DataFrame, sensors: list, n_bins: int = 10) -> dict:
    """
    For every sensor: the bin edges (training deciles) and the fraction of
    training readings in each bin. Saved as JSON next to the model.
    """
    baseline = {}
    for sensor in sensors:
        values = train[sensor].to_numpy(dtype=float)
        # Inner cut points at 10%, 20%, ..., 90%. np.unique removes repeated
        # edges: sensor s17 only takes a few whole-number values, so some
        # deciles are equal and those bins would be empty anyway.
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(values, quantiles))
        expected = bin_fractions(values, edges)
        baseline[sensor] = {"edges": edges.round(6).tolist(),
                            "expected": expected.round(6).tolist()}
    return baseline


def bin_fractions(values: np.ndarray, edges) -> np.ndarray:
    """Fraction of `values` in each bin. With k inner edges there are k + 1 bins."""
    # searchsorted gives each value the index of the bin it falls into.
    # side="right" puts a value equal to an edge in the bin above it.
    bin_index = np.searchsorted(np.asarray(edges), values, side="right")
    counts = np.bincount(bin_index, minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(expected, actual) -> float:
    """Population Stability Index between two lists of bin fractions."""
    expected = np.clip(np.asarray(expected, dtype=float), EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_per_sensor(baseline: dict, readings: pd.DataFrame) -> dict:
    """PSI of every sensor in the baseline for a batch of readings."""
    return {sensor: psi(b["expected"], bin_fractions(readings[sensor].to_numpy(float), b["edges"]))
            for sensor, b in baseline.items()}


# ----------------------------------------------------------------- monitor
class DriftMonitor:
    """
    Keeps the most recent `window_rows` sensor readings received by the API
    and reports their PSI against the baseline.

    A class because it keeps state between requests (the window of recent
    rows). A lock protects that window, because FastAPI can run two
    requests at the same time.
    """

    def __init__(self, sensors: list, threshold: float = 1.0,
                 min_rows: int = 500, window_rows: int = 1000):
        self.sensors = list(sensors)
        self.threshold = threshold
        self.min_rows = min_rows
        self.baseline = None                    # set by set_baseline()
        self.baseline_version = None            # model version the baseline came from
        self._rows = deque(maxlen=window_rows)  # oldest rows fall out automatically
        self._lock = threading.Lock()

    @property
    def window_rows(self) -> int:
        return self._rows.maxlen

    def set_baseline(self, baseline: dict, model_version: str) -> None:
        """
        Use a new baseline (when a new model is loaded). The window is
        cleared: readings judged against the old training data should not
        decide whether the new model sees drift.
        """
        with self._lock:
            self.baseline = {s: baseline[s] for s in self.sensors if s in baseline}
            self.baseline_version = model_version
            self._rows.clear()

    def add(self, readings: pd.DataFrame) -> None:
        """Remember the sensor values of every received row."""
        values = readings[self.sensors].to_numpy(dtype=float)
        with self._lock:
            self._rows.extend(values)

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()

    def status(self) -> dict:
        """
        Current drift state. Possible states:
            no_baseline     the loaded model has no baseline (trained before Task 8)
            collecting      fewer than min_rows readings so far: PSI not trusted yet
            ok / drift      max PSI below / above the threshold
        """
        with self._lock:
            rows = np.array(self._rows)
        n_rows = len(rows)
        result = {"state": "no_baseline", "n_rows": n_rows, "min_rows": self.min_rows,
                  "window_rows": self.window_rows, "threshold": self.threshold,
                  "baseline_model_version": self.baseline_version,
                  "max_psi": None, "worst_sensor": None, "drift_detected": False,
                  "sensor_psi": {}}
        if self.baseline is None:
            return result
        if n_rows < self.min_rows:
            result["state"] = "collecting"
            return result

        frame = pd.DataFrame(rows, columns=self.sensors)
        scores = psi_per_sensor(self.baseline, frame)
        worst = max(scores, key=scores.get)
        drift = scores[worst] > self.threshold
        result.update({"state": "drift" if drift else "ok",
                       "max_psi": round(scores[worst], 4), "worst_sensor": worst,
                       "drift_detected": drift,
                       "sensor_psi": {s: round(v, 4) for s, v in scores.items()}})
        return result
