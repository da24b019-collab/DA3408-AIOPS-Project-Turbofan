"""
Stage 3 of the pipeline: train a model and track everything in MLflow.

Steps:
    1. Load the Spark features for the training subset (FD001).
    2. Validate with GroupKFold: each engine is entirely in train or in
       validation, never both (a random row split leaks, as measured earlier).
    3. Choose the decision threshold on the validation predictions so that
       at least `target_recall` of real failures are caught.
    4. Train the final model on all training engines (with MLflow autolog on).
    5. Evaluate on the official NASA test set (last cycle of each engine).
    6. Log to MLflow: parameters, metrics, plots, SHAP explanation, the Git
       commit, the DVC data version and the dataset itself. Also the drift
       baseline: how the training readings of each sensor are spread out,
       which the API later compares new readings against.
    7. Register the model as a new version with the alias "challenger".
       A human later promotes it to "champion" (src/models/promote.py).

Run it with:
    python -m src.models.train                       # model type from params.yaml
    python -m src.models.train --model-type random_forest
"""

import argparse
import json
import shutil
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, SENSOR_DESCRIPTIONS, load_params
from src.features.feature_logic import feature_names
from src.logger import get_logger
from src.models.evaluate import (
    choose_threshold,
    classification_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
)
from src.models.tracking import check_mlflow_server, dvc_output_hash, git_commit, git_is_dirty
from src.monitoring.drift import build_baseline
from src.monitoring.pipeline_metrics import push_stage_metrics

logger = get_logger(__name__)

MODEL_TYPES = ["xgboost", "random_forest", "logistic_regression"]
REPORT_PATH = PROJECT_ROOT / "reports" / "train_metrics.json"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "latest"
ARTIFACT_DIR = PROJECT_ROOT / "tmp" / "train_artifacts"
# MLflow saves scikit-learn models with "skops", a format that refuses to load
# unknown classes (a pickled file can otherwise run any code when opened).
# XGBoost's classes are not on skops' built-in list, so we trust them by name.
TRUSTED_TYPES = {
    "xgboost": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
    "random_forest": ["sklearn.tree._tree.Tree"],
}


def build_model(model_type: str, cfg: dict, pos_weight: float, seed: int):
    """Create an untrained model. Every model handles class imbalance in its own way."""
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            scale_pos_weight=pos_weight,  # weight "at risk" rows by (safe / at-risk)
            n_jobs=-1, random_state=seed, eval_metric="logloss")
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"],
            class_weight="balanced", n_jobs=-1, random_state=seed)
    if model_type == "logistic_regression":
        # Scaling matters for logistic regression because sensors have very different units
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=cfg["C"], class_weight="balanced", max_iter=2000))
    raise ValueError(f"Unknown model_type '{model_type}'. Choose from {MODEL_TYPES}")


def load_split(input_dir, split: str, subset: str) -> pd.DataFrame:
    path = PROJECT_ROOT / input_dir / f"{split}_{subset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dvc repro features` first.")
    return pd.read_parquet(path).sort_values(["unit", "cycle"]).reset_index(drop=True)


def cross_validate(model_type, cfg, X, y, groups, n_splits, seed):
    """
    Out-of-fold predictions: every training row gets a probability from a
    model that never saw its engine. Choosing the threshold on these is fair.
    """
    oof_prob = np.zeros(len(y))
    fold_pr_auc = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        pos_weight = (y.iloc[tr] == 0).sum() / (y.iloc[tr] == 1).sum()
        model = build_model(model_type, cfg, pos_weight, seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        oof_prob[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_pr_auc.append(classification_metrics(y.iloc[va], oof_prob[va], 0.5)["pr_auc"])
        logger.info("Fold %d: PR-AUC %.3f", fold, fold_pr_auc[-1])
    return oof_prob, fold_pr_auc


def feature_importance(model, model_type: str, features: list) -> pd.DataFrame:
    if model_type == "logistic_regression":
        values = np.abs(model[-1].coef_[0])  # size of each (scaled) coefficient
    else:
        values = model.feature_importances_
    imp = pd.DataFrame({"feature": features, "importance": values})
    imp["sensor_name"] = imp["feature"].str.split("_").str[0].map(SENSOR_DESCRIPTIONS)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def save_importance_plot(imp: pd.DataFrame, path) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([f"{f} ({n})" for f, n in zip(top["feature"], top["sensor_name"])],
            top["importance"], color="steelblue")
    ax.set_title("Top 15 features by importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_shap_plot(model, model_type: str, X_sample: pd.DataFrame, path) -> bool:
    """SHAP summary: which features push predictions towards 'at risk' and how strongly."""
    if model_type == "logistic_regression":
        return False  # tree explainer does not apply; importance plot covers this model
    values = shap.TreeExplainer(model).shap_values(X_sample)
    if isinstance(values, list):      # older SHAP: one array per class
        values = values[1]
    elif np.ndim(values) == 3:        # newer SHAP for random forest: (rows, features, classes)
        values = values[:, :, 1]
    shap.summary_plot(values, X_sample, show=False, max_display=15)
    plt.gcf().set_size_inches(9, 7)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")
    return True


def enable_autolog(model_type: str, enable: bool) -> None:
    """Autolog records the model's own parameters and training metrics automatically."""
    if model_type == "xgboost":
        mlflow.xgboost.autolog(disable=not enable, log_models=False, log_datasets=False, silent=True)
    else:
        mlflow.sklearn.autolog(disable=not enable, log_models=False, log_datasets=False, silent=True)


def main(model_type_override: str | None = None) -> dict:
    params = load_params()
    tcfg, mcfg = params["train"], params["mlflow"]
    feat = params["features"]
    model_type = model_type_override or tcfg["model_type"]
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {MODEL_TYPES}")
    seed = int(tcfg["random_state"])
    features = feature_names(feat["sensors"], int(feat["window"]))

    check_mlflow_server(mcfg["tracking_uri"])
    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split(tcfg["input_dir"], "train", tcfg["subset"])
    test = load_split(tcfg["input_dir"], "test", tcfg["subset"])
    # All features as decimals: one sensor (s17) holds whole numbers, and an
    # integer column in the model's input schema would reject decimal input later.
    train[features] = train[features].astype("float64")
    test[features] = test[features].astype("float64")
    X, y, groups = train[features], train["label"], train["unit"]
    test_last = test.groupby("unit").tail(1)  # official NASA evaluation point
    logger.info("Training %s on %s: %d rows, %d engines, %d features",
                model_type, tcfg["subset"], len(train), groups.nunique(), len(features))

    with mlflow.start_run(run_name=f"{model_type}-{tcfg['subset']}") as run:
        # ---- Where this run came from (reproducibility) ----
        mlflow.set_tags({
            "model_type": model_type,
            "git_commit": git_commit(),
            "git_dirty": str(git_is_dirty()),
            "dvc_features_md5": dvc_output_hash("features", "data/features"),
            "train_subset": tcfg["subset"],
        })
        mlflow.log_params({
            "model_type": model_type, "subset": tcfg["subset"], "n_splits": tcfg["n_splits"],
            "target_recall": tcfg["target_recall"], "window": feat["window"],
            "n_features": len(features), "sensors": ",".join(feat["sensors"]),
        })
        with warnings.catch_warnings():
            # The 0/1 label column is an integer; MLflow warns about that, but it is correct here.
            warnings.simplefilter("ignore", UserWarning)
            mlflow.log_input(
                mlflow.data.from_pandas(train[features + ["label"]], targets="label",
                                        name=f"train_{tcfg['subset']}_features"),
                context="training")

        # ---- 1. Cross-validation and threshold ----
        start = time.perf_counter()
        oof_prob, fold_pr_auc = cross_validate(
            model_type, tcfg[model_type], X, y, groups, int(tcfg["n_splits"]), seed)
        threshold = choose_threshold(y, oof_prob, float(tcfg["target_recall"]))
        cv = classification_metrics(y, oof_prob, threshold)
        cv_default = classification_metrics(y, oof_prob, 0.5)
        logger.info("Chosen threshold %.3f -> CV recall %.3f, precision %.3f",
                    threshold, cv["recall"], cv["precision"])

        # ---- 2. Final model on all training engines (autolog on) ----
        pos_weight = (y == 0).sum() / (y == 1).sum()
        model = build_model(model_type, tcfg[model_type], pos_weight, seed)
        enable_autolog(model_type, True)
        model.fit(X, y)
        enable_autolog(model_type, False)
        train_seconds = time.perf_counter() - start

        # ---- 3. Official test set ----
        test_prob = model.predict_proba(test_last[features])[:, 1]
        test_scores = classification_metrics(test_last["label"], test_prob, threshold)

        metrics = {"threshold": threshold, "train_seconds": round(train_seconds, 2),
                   "cv_pr_auc_fold_std": float(np.std(fold_pr_auc)),
                   "test_always_safe_accuracy": float(1 - test_last["label"].mean())}
        metrics.update({f"cv_{k}": v for k, v in cv.items()})
        metrics.update({f"cv_default05_{k}": v for k, v in cv_default.items()})
        metrics.update({f"test_{k}": v for k, v in test_scores.items()})
        mlflow.log_metrics(metrics)

        # ---- 4. Artifacts: plots and tables ----
        plot_pr_curve(y, oof_prob, threshold, ARTIFACT_DIR / "pr_curve_cv.png",
                      f"{model_type}: validation precision-recall")
        plot_confusion_matrix(test_last["label"], test_prob, threshold,
                              ARTIFACT_DIR / "confusion_matrix_test.png",
                              f"{model_type}: test set (100 engines)")
        imp = feature_importance(model, model_type, features)
        imp.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
        save_importance_plot(imp, ARTIFACT_DIR / "feature_importance.png")
        sample = X.sample(500, random_state=seed)
        has_shap = save_shap_plot(model, model_type, sample, ARTIFACT_DIR / "shap_summary.png")
        for name in ["pr_curve_cv.png", "confusion_matrix_test.png",
                     "feature_importance.csv", "feature_importance.png"] + (
                        ["shap_summary.png"] if has_shap else []):
            mlflow.log_artifact(str(ARTIFACT_DIR / name), artifact_path="evaluation")

        # Drift baseline: decile bins of each RAW sensor in the training data.
        # Stored in the same run as the model, so every model version carries
        # the description of the data it was trained on.
        baseline = build_baseline(train, feat["sensors"], int(params["monitoring"]["psi_bins"]))
        mlflow.log_dict(baseline, "drift/baseline.json")

        # ---- 5. Log and register the model ----
        # pyfunc_predict_fn="predict_proba": when served, the model returns
        # probabilities, and the API applies the threshold stored with it.
        info = mlflow.sklearn.log_model(
            sk_model=model, name="model",
            signature=infer_signature(X.head(5), model.predict_proba(X.head(5))),
            input_example=X.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=TRUSTED_TYPES.get(model_type),
            registered_model_name=mcfg["registered_model_name"],
            metadata={"threshold": threshold, "features": features,
                      "window": feat["window"], "model_type": model_type})
        version = info.registered_model_version
        client = MlflowClient()
        client.set_registered_model_alias(mcfg["registered_model_name"], "challenger", version)
        for key, value in {"threshold": f"{threshold:.6f}", "model_type": model_type,
                           "test_recall": f"{test_scores['recall']:.3f}",
                           "test_pr_auc": f"{test_scores['pr_auc']:.3f}",
                           "git_commit": git_commit()}.items():
            client.set_model_version_tag(mcfg["registered_model_name"], version, key, value)

    # ---- 6. Local copy for DVC versioning, and a small report ----
    if LOCAL_MODEL_DIR.exists():
        shutil.rmtree(LOCAL_MODEL_DIR)
    mlflow.sklearn.save_model(model, str(LOCAL_MODEL_DIR), pyfunc_predict_fn="predict_proba",
                              skops_trusted_types=TRUSTED_TYPES.get(model_type),
                              metadata={"threshold": threshold, "features": features})
    report = {"run_id": run.info.run_id, "model_type": model_type, "model_version": version,
              **{k: round(float(v), 4) for k, v in metrics.items()}}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("Run %s | model version %s (alias: challenger)", run.info.run_id, version)
    logger.info("Test: recall %.3f, precision %.3f, PR-AUC %.3f",
                test_scores["recall"], test_scores["precision"], test_scores["pr_auc"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and track a failure classifier")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=None,
                        help="overrides train.model_type in params.yaml")
    args = parser.parse_args()

    # Run the stage and report the outcome to Prometheus (via the Pushgateway),
    # so a failed training run shows up on the Grafana dashboard too.
    started = time.perf_counter()
    try:
        result = main(args.model_type)
    except Exception:
        push_stage_metrics("train", time.perf_counter() - started, success=False)
        raise
    push_stage_metrics("train", time.perf_counter() - started, success=True,
                       values={"model_version": result["model_version"],
                               "test_recall": result["test_recall"],
                               "test_precision": result["test_precision"],
                               "test_pr_auc": result["test_pr_auc"],
                               "threshold": result["threshold"]})
