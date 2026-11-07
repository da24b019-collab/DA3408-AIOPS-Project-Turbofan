# Model Training Findings (MLflow)

## Setup
- Data: Spark features for FD001 (100 engines, 20,631 rows, 42 features).
- Validation: GroupKFold by engine, 5 folds (no engine in both train and validation).
- Threshold rule: the highest threshold whose out-of-fold recall is at least 0.90.
- Test: official NASA test set, last cycle of each of 100 engines (25 truly at risk).
- Tracking: MLflow 3.18 server (SQLite backend), experiment "turbofan-failure-prediction".

## Results (same data, folds and threshold rule for every model)

| Model | Threshold | CV precision (recall 0.90) | Test recall | Test precision | Test PR-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.531 | 0.782 | 0.80 (20/25) | 0.870 (3 false alarms) | 0.938 |
| Random forest | 0.546 | 0.811 | 0.80 (20/25) | 0.952 (1 false alarm) | 0.941 |
| Logistic regression | 0.723 | 0.821 | 0.76 (19/25) | 0.864 (3 false alarms) | 0.941 |

XGBoost fold PR-AUC: 0.927, 0.961, 0.941, 0.963, 0.952 (consistent across engine groups).

## Serving cost (experiments/model_cost_benchmark.py, measured on this laptop)

| Model | Size | Predict one engine | SHAP for one row |
|---|---|---|---|
| XGBoost | 465 KB | 2.9 ms | 3.7 ms |
| Random forest | 8,113 KB | 65.9 ms | 23.6 ms |
| Logistic regression | 3 KB | 2.5 ms | ~0 ms |

## Decision: XGBoost
- Quality is statistically tied: differences are 1 to 2 engines on a 25-engine test set.
- Random forest ruled out for serving: 17x larger, 23x slower to predict, 6x slower to explain.
- Logistic regression is competitive, but retraining on mixed operating conditions (FD002/FD004)
  needs non-linear patterns, which a linear model cannot learn. EDA also showed s9 and s14
  matter in non-linear ways.

## What MLflow records per run (beyond autolog)
- Autolog: the model's own parameters and training metrics (final fit only).
- Manual: git_commit, git_dirty, dvc_features_md5, the training dataset, ~22 metrics
  (CV, CV at default 0.5, test, always-safe baseline), and artifacts: PR curve,
  confusion matrix, feature importance table and plot, SHAP summary.

## Model registry and approval
- Every run registers a new version with the alias "challenger".
- A human compares and approves with `python -m src.models.promote --approve`,
  which moves the alias "champion" (served by the API). Rollback: `--rollback VERSION`.
- Version 6 (XGBoost, trained by the DVC pipeline from commit 0b9de917) is the champion.
  Versions 5 and 6 have identical scores from different commits: reproducible results.
