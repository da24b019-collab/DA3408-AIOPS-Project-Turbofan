"""
Human approval step for new models (and rollback).

Every training run registers a new model version with the alias
"challenger". The API serves whichever version has the alias "champion".
A model only becomes champion when a person decides so with this script,
after comparing the two.

    python -m src.models.promote                  # compare champion and challenger
    python -m src.models.promote --approve        # challenger becomes champion
    python -m src.models.promote --rollback 3     # make version 3 champion again

--approve applies the same approval gate as the dashboard's Approve button
(src/models/compare.py): both models are scored on the same NASA test sets,
and the challenger is refused if it is clearly worse on FD001. Since Task 9
the usual way is the dashboard, which also restarts the model server through
the retraining worker; after using this script, restart it yourself:
    docker compose restart model-server
"""

import argparse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import load_params
from src.models.compare import compare_versions
from src.logger import get_logger
from src.models.tracking import check_mlflow_server

logger = get_logger(__name__)
SHOWN_TAGS = ["model_type", "train_subsets", "threshold", "test_recall", "git_commit"]


def get_version(client, name: str, alias: str):
    """The model version holding this alias, or None if no version has it yet."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def describe(label: str, mv) -> None:
    if mv is None:
        print(f"{label:<11}: none")
        return
    tags = ", ".join(f"{t}={mv.tags.get(t, '-')[:10]}" for t in SHOWN_TAGS)
    print(f"{label:<11}: version {mv.version} | {tags}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve or roll back the served model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--approve", action="store_true", help="promote challenger to champion")
    group.add_argument(
        "--rollback", type=str, metavar="VERSION", help="make an earlier version champion"
    )
    args = parser.parse_args()

    cfg = load_params()["mlflow"]
    check_mlflow_server(cfg["tracking_uri"])
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    client = MlflowClient()
    name = cfg["registered_model_name"]

    champion = get_version(client, name, "champion")
    challenger = get_version(client, name, "challenger")
    describe("Champion", champion)
    describe("Challenger", challenger)

    if args.approve:
        if challenger is None:
            raise SystemExit("No challenger to approve. Train a model first.")
        if champion is not None and champion.version != challenger.version:
            max_drop = float(load_params()["retraining"]["max_fd001_drop"])
            result = compare_versions(name, champion.version, challenger.version, max_drop)
            for subset, scores in result["scores"]["challenger"].items():
                old = result["scores"]["champion"][subset]
                print(
                    f"  {subset}: recall {old['recall']:.3f} -> {scores['recall']:.3f}, "
                    f"PR-AUC {old['pr_auc']:.3f} -> {scores['pr_auc']:.3f}"
                )
            for check in result["checks"]:
                print(
                    f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}"
                )
            if not result["approvable"]:
                raise SystemExit("Approval refused: a check failed.")
        client.set_registered_model_alias(name, "champion", challenger.version)
        logger.info("Approved: version %s is now champion", challenger.version)
    elif args.rollback:
        client.get_model_version(name, args.rollback)  # fails clearly if it does not exist
        client.set_registered_model_alias(name, "champion", args.rollback)
        logger.info("Rolled back: version %s is now champion", args.rollback)


if __name__ == "__main__":
    main()
