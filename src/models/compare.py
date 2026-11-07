"""
Champion vs challenger: compare two model versions on the SAME test data,
and decide whether the challenger may replace the champion.

Each model is scored on the official NASA test engines of every dataset
(last cycle of each engine), with its own decision threshold:
    FD001  the original task (1 flight condition, 1 fault)
    FD002  6 flight conditions          FD003  2 faults
    FD004  6 flight conditions, 2 faults

Comparing on the same data with the same code is the only fair comparison:
the scores stored with each run may come from different test sets (a model
trained on FD001+FD002 reports its test score on both).

Used by the API (GET /model/compare, POST /model/approve) and by
src/models/promote.py.
"""

import mlflow
import pandas as pd

from src.config import PROJECT_ROOT
from src.models.evaluate import classification_metrics

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def load_test_engines(features_dir, subset: str, features: list) -> pd.DataFrame:
    """Last cycle of each test engine: the official evaluation point."""
    path = features_dir / f"test_{subset}.parquet"
    test = pd.read_parquet(path).sort_values(["unit", "cycle"])
    test[features] = test[features].astype("float64")
    return test.groupby("unit").tail(1)


def evaluate_on_test_sets(
    model, threshold: float, features: list, features_dir=None, subsets=ALL_SUBSETS
) -> dict:
    """{subset: {"recall": .., "precision": .., "pr_auc": .., ...}} for one model."""
    features_dir = features_dir or PROJECT_ROOT / "data" / "features"
    scores = {}
    for subset in subsets:
        test = load_test_engines(features_dir, subset, features)
        prob = model.predict_proba(test[features])[:, 1]
        scores[subset] = {
            k: round(float(v), 4)
            for k, v in classification_metrics(test["label"], prob, threshold).items()
        }
    return scores


def load_version(model_name: str, version: str):
    """The model object and its saved threshold and feature list."""
    uri = f"models:/{model_name}/{version}"
    metadata = mlflow.models.get_model_info(uri).metadata or {}
    return mlflow.sklearn.load_model(uri), float(metadata["threshold"]), list(metadata["features"])


def promotion_checks(champion_scores: dict, challenger_scores: dict, max_drop: float) -> list:
    """
    The approval gate. Every check must pass before a challenger can become
    champion. Returns [{"name", "passed", "detail"}, ...].

    The rule protects the ORIGINAL task: a model retrained for new flight
    conditions must still detect FD001 failures about as well as before.
    """
    checks = []
    for metric in ["recall", "pr_auc"]:
        old, new = champion_scores["FD001"][metric], challenger_scores["FD001"][metric]
        checks.append(
            {
                "name": f"FD001 {metric} not worse by more than {max_drop:.2f}",
                "passed": new >= old - max_drop,
                "detail": f"champion {old:.3f}, challenger {new:.3f}",
            }
        )
    return checks


def compare_versions(
    model_name: str, champion: str, challenger: str, max_drop: float, features_dir=None
) -> dict:
    """Score both versions on all test sets and run the approval checks."""
    result = {"champion_version": champion, "challenger_version": challenger, "scores": {}}
    for role, version in [("champion", champion), ("challenger", challenger)]:
        model, threshold, features = load_version(model_name, version)
        result["scores"][role] = evaluate_on_test_sets(model, threshold, features, features_dir)
    result["checks"] = promotion_checks(
        result["scores"]["champion"], result["scores"]["challenger"], max_drop
    )
    result["approvable"] = (champion != challenger) and all(c["passed"] for c in result["checks"])
    return result
