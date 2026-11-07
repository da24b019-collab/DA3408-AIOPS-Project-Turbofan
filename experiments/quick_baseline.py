"""
Preliminary experiment: compare three simple models on FD001.

This is NOT the final training stage (that comes later, with MLflow tracking).
It answers two early design questions:
    1. Does the choice of model matter much on this dataset?
    2. How much does a leaky random split inflate the scores?

Run from the project root, after `dvc repro`:
    python -m experiments.quick_baseline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, SENSOR_COLUMNS

DATA_DIR = PROJECT_ROOT / "data" / "interim"
WINDOW = 10  # number of past cycles used for the rolling features


def load_fd001():
    """Load the train and test tables written by the ingest stage."""
    train = pd.read_parquet(DATA_DIR / "train_FD001.parquet")
    test = pd.read_parquet(DATA_DIR / "test_FD001.parquet")
    return train, test


def pick_useful_sensors(train):
    """Keep only sensors that actually change. A flat sensor tells the model nothing."""
    stds = train[SENSOR_COLUMNS].std()
    useful = [s for s in SENSOR_COLUMNS if stds[s] > 1e-4]
    dropped = [s for s in SENSOR_COLUMNS if s not in useful]
    return useful, dropped


def add_rolling_features(df, sensors):
    """
    For each engine separately, add the mean and standard deviation of the last
    WINDOW cycles. A single reading is noisy; the recent average and spread show
    the trend. Grouping by unit stops one engine's values leaking into another's.
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")[sensors]
    roll_mean = grouped.rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    # std of a single value is undefined (NaN), so the first cycle gets 0
    roll_std = grouped.rolling(WINDOW, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    for s in sensors:
        df[f"{s}_mean{WINDOW}"] = roll_mean[s]
        df[f"{s}_std{WINDOW}"] = roll_std[s]
    return df


def score(y_true, prob, threshold=0.5):
    """Metrics for an imbalanced problem. Accuracy is included only for comparison."""
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, prob),
    }


def main():
    train, test = load_fd001()
    sensors, dropped = pick_useful_sensors(train)
    print("Dropped flat sensors:", dropped)

    train = add_rolling_features(train, sensors)
    test = add_rolling_features(test, sensors)
    features = sensors + [f"{s}_mean{WINDOW}" for s in sensors] + [f"{s}_std{WINDOW}" for s in sensors]
    print("Number of features:", len(features))

    X, y, groups = train[features], train["label"], train["unit"]
    # XGBoost's way of handling imbalance: weight positives by (negatives / positives)
    pos_weight = (y == 0).sum() / (y == 1).sum()

    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=42),
    }

    # The official NASA test point is the LAST recorded cycle of each test engine.
    test_last = test.groupby("unit").tail(1)
    print(f"Test engines: {len(test_last)}, at risk: {int(test_last['label'].sum())}")

    cv_rows, test_rows = {}, {}
    for name, model in models.items():
        # GroupKFold: each engine is entirely in train OR validation, never both.
        fold_scores = []
        for tr_idx, va_idx in GroupKFold(n_splits=5).split(X, y, groups):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            fold_scores.append(score(y.iloc[va_idx], model.predict_proba(X.iloc[va_idx])[:, 1]))
        cv_rows[name] = pd.DataFrame(fold_scores).mean()

        model.fit(X, y)
        test_rows[name] = score(test_last["label"], model.predict_proba(test_last[features])[:, 1])

    # A fake model that always says "safe": shows why accuracy is misleading here.
    cv_rows["Always say safe"] = score(y, np.zeros(len(y)))
    test_rows["Always say safe"] = score(test_last["label"], np.zeros(len(test_last)))

    print("\n=== Validation: GroupKFold by engine (mean of 5 folds) ===")
    print(pd.DataFrame(cv_rows).T.round(3))
    print("\n=== Official test set: last cycle of each engine ===")
    print(pd.DataFrame(test_rows).T.round(3))

    # Leakage demo: a random row split puts neighbouring cycles of the same
    # engine in both train and validation, so the model is tested on near-copies.
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    leaky = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          scale_pos_weight=pos_weight, n_jobs=-1, random_state=42)
    leaky.fit(X_tr, y_tr)
    print("\n=== XGBoost with a LEAKY random row split (do not use) ===")
    print({k: round(v, 3) for k, v in score(y_va, leaky.predict_proba(X_va)[:, 1]).items()})


if __name__ == "__main__":
    main()
