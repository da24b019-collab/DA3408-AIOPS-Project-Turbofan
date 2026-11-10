# Low-Level Design: Pipeline, Model, Monitoring and Retraining

This document describes the modules outside the REST API. The API (every endpoint, its
request and response, errors and its internal classes) is specified in `docs/lld_api.md`.
The reasons behind the choices are in `docs/hld.md`.

## 1. Code layout

```
src/
  config.py              column names, sensor descriptions, load_params()
  logger.py              get_logger(): console + rotating files in logs/
  data/ingest.py         pipeline stage 1
  features/
    feature_logic.py     feature definition shared by training and serving (pandas)
    build_features.py    pipeline stage 2 (Spark)
  models/
    train.py             pipeline stage 3
    evaluate.py          metrics, threshold rule, plots
    tracking.py          git commit / dirty flag, DVC output hash, MLflow server check
    compare.py           champion vs challenger, approval checks
    promote.py           command-line approval and rollback
  monitoring/
    drift.py             PSI baselines and DriftMonitor
    pipeline_metrics.py  push stage metrics to the Pushgateway
  retraining/worker.py   runs retrain and deploy jobs
  api/                   the backend (see lld_api.md)
frontend/                the Streamlit dashboard (see section 8)
scripts/                 make_samples.py, latency_test.py
experiments/             benchmarks and studies behind the design decisions
```

## 2. Configuration (`params.yaml`)

| Section | Keys | Used by |
|---|---|---|
| `data` | `raw_dir`, `interim_dir`, `subsets` | ingest |
| `label` | `horizon: 30` | ingest |
| `features` | `sensors` (14), `window: 10`, input and output folders | features, train, API |
| `spark` | `master: local[*]`, `driver_memory: 4g`, `shuffle_partitions: 12`, `local_dir` | features |
| `train` | `subsets`, `model_type`, `n_splits: 5`, `target_recall: 0.90`, `random_state: 42`, model hyper-parameters | train |
| `mlflow` | `tracking_uri`, `experiment_name`, `registered_model_name` | train, promote, API |
| `monitoring` | `psi_bins: 10`, `drift_threshold: 1.0`, `min_rows: 3000`, `window_rows: 5000`, `pushgateway_url` | train, API |
| `retraining` | `subsets_on_drift`, `cooldown_minutes: 30`, `max_fd001_drop: 0.05` | API, compare |

DVC lists the sections each stage depends on, so changing a value reruns only the
stages that use it.

## 3. Pipeline stages (`dvc.yaml`)

```
data/raw (DVC) ──ingest──> data/interim ──features──> data/features ──train──> models/latest
                                                                          └──> MLflow run + model version
```

### 3.1 Stage `ingest` (`src/data/ingest.py`)

| | |
|---|---|
| Input | `data/raw/train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` (space-separated, no header) |
| Output | `data/interim/{train,test}_FD00x.parquet`; `reports/ingest_report.json` |
| Output columns | `unit, cycle, setting_1..3, s1..s21, rul, label, subset` |

| Function | What it does |
|---|---|
| `read_cmapss_file(path)` | Reads one file, names the 26 columns; raises `DataValidationError` on a wrong column count |
| `read_rul_file(path)` | Reads the true remaining life of each test engine |
| `validate_frame(df, name)` | No missing values; unit and cycle numbers at least 1; cycles of each engine run 1, 2, 3, ... with no gaps |
| `add_train_labels(df, horizon)` | `rul = last cycle − cycle`; `label = rul ≤ horizon` (inclusive) |
| `add_test_labels(df, true_rul, horizon)` | Same, using the true RUL at the end of each test engine |
| `ingest_subset(...)`, `summarise(df)`, `main()` | Runs the above per dataset and writes the report |

### 3.2 Stage `features` (`src/features/build_features.py`, `feature_logic.py`)

| | |
|---|---|
| Input | `data/interim/*.parquet` |
| Output | `data/features/*.parquet` (interim columns + 28 new); `reports/features_report.json`; Pushgateway metrics |
| Features (42, fixed order) | `s2 … s21` (14 raw), `s2_mean10 … s21_mean10` (14), `s2_std10 … s21_std10` (14) |

Rules followed by both implementations (Spark for training, pandas for serving):
per engine, in cycle order; mean and sample standard deviation over the last 10 cycles,
using the cycles that exist (cycle 3 averages cycles 1 to 3); standard deviation of a
single value is 0.

| Function | What it does |
|---|---|
| `create_spark_session(params)` | Local Spark with the memory and partition settings from `spark` |
| `add_rolling_features(sdf, sensors, window)` | Spark `Window.partitionBy("subset", "unit").orderBy("cycle").rowsBetween(-(window−1), 0)` with `avg` and `stddev_samp` |
| `process_table(...)` | Read one parquet, add features, write it, return row count and time |
| `feature_names(sensors, window)` | The 42 names in the fixed order |
| `compute_features_pandas(df, sensors, window)` | The same features with pandas `groupby("unit").rolling(window, min_periods=1)` |
| `main()` | All tables; report with rows per second; `push_stage_metrics("features", ...)` |

### 3.3 Stage `train` (`src/models/train.py`)

| | |
|---|---|
| Input | `data/features/train_*.parquet` and `test_*.parquet` for `train.subsets` |
| Output | `models/latest/` (local copy of the trained model in MLflow format, versioned by DVC); `reports/train_metrics.json`; MLflow run; new registered version with alias **challenger**; Pushgateway metrics |

Steps of `main()`:

1. Check the MLflow server answers (`check_mlflow_server`).
2. Load and join the training subsets; engine id = `subset + "_" + unit` (unit numbers repeat across datasets).
3. `cross_validate`: 5-fold GroupKFold by engine; out-of-fold probabilities.
4. `choose_threshold(y, prob, 0.90)`: the highest threshold with out-of-fold recall ≥ 0.90.
5. Fit on all training rows; score the last cycle of every test engine of the training subsets.
6. `evaluate_on_test_sets`: also score FD001 to FD004 test sets separately (metrics `test_FD00x_recall`, `test_FD00x_pr_auc`).
7. Log to MLflow: parameters, metrics, PR curve, confusion matrix, feature importance, SHAP summary, dataset, tags (`git_commit`, `git_dirty`, `dvc_features_md5`, `train_subsets`), and `drift/baseline.json`.
8. Register the model (with `pyfunc_predict_fn="predict_proba"` and the threshold and feature list as metadata); set alias `challenger`.
9. Write `reports/train_metrics.json`; push stage metrics (success or failure) to the Pushgateway.

Drift baseline format (`drift/baseline.json`):
`{"FD001": {"s2": {"edges": [9 inner decile edges], "expected": [10 fractions]}, ...}, "FD002": {...}}`

### 3.4 Evaluation (`src/models/evaluate.py`)

| Function | Returns |
|---|---|
| `choose_threshold(y_true, prob, target_recall)` | Threshold (float) |
| `classification_metrics(y_true, prob, threshold)` | `precision, recall, f1, pr_auc, roc_auc, accuracy` (accuracy kept only for comparison) |
| `plot_pr_curve`, `plot_confusion_matrix` | PNG files logged to MLflow |

## 4. Champion vs challenger (`src/models/compare.py`)

| Function | What it does |
|---|---|
| `load_test_engines(dir, subset, features)` | Last cycle of each test engine of one dataset |
| `evaluate_on_test_sets(model, threshold, features, dir, subsets)` | `{subset: metrics}` for FD001 to FD004 |
| `load_version(name, version)` | Model object, its threshold and feature list from the registry |
| `promotion_checks(champion, challenger, max_drop)` | `[{"name", "passed", "detail"}]`: FD001 recall and PR-AUC each must not drop by more than `max_drop` |
| `compare_versions(name, champion, challenger, max_drop)` | Scores of both, the checks, and `approvable` |

`promote.py` (command line) applies the same checks before `--approve`; `--rollback N`
moves the champion alias without checks.

## 5. Drift detection (`src/monitoring/drift.py`)

```
PSI = Σ over bins (actual − expected) · ln(actual / expected),   empty bins → 0.0001
```

| Function / class | What it does |
|---|---|
| `build_baseline(train, sensors, n_bins)` | Per sensor: inner decile edges (duplicates removed) and the fraction of training rows per bin |
| `bin_fractions(values, edges)` | Fractions per bin with `searchsorted` |
| `psi(expected, actual)` | The formula above |
| `as_named_baselines(saved)` | Accepts the old single-baseline format (wraps it as `"training"`) |
| `psi_per_sensor(baseline, readings)` | `{sensor: PSI}` |
| `DriftMonitor` | Holds a `deque(maxlen=window_rows)` of recent readings and a lock |

`DriftMonitor` methods: `set_baseline(baseline, version)` (also empties the window),
`add(readings)`, `reset()`, `status()`.
`status()` states: `no_baseline` → `collecting` (fewer than `min_rows`) → `ok` / `drift`.
With several baselines, the one with the lowest maximum PSI (`closest_baseline`) is used.

## 6. Pipeline metrics (`src/monitoring/pipeline_metrics.py`)

`push_stage_metrics(stage, seconds, success, values)` pushes, under job `pdm_<stage>`:
`pdm_pipeline_stage_duration_seconds`, `pdm_pipeline_stage_success`,
`pdm_pipeline_stage_last_run_timestamp` and `pdm_<stage>_<name>` for each value. A
connection error is logged as a warning and returns `False`: monitoring never stops the pipeline.

## 7. Retraining worker (`src/retraining/worker.py`)

Loop: every 10 s `POST /jobs/claim`; do the job; `POST /jobs/{id}/finish`. `--once` does one job.

**Retrain job** (`do_retrain`):

1. Refuse if `git status --porcelain --untracked-files=no` is not empty.
2. Remember `HEAD`. Set `train.subsets` in params.yaml with `set_train_subsets` (line edit inside the `train:` section only, comments kept); commit if changed.
3. `dvc repro -f -s train`.
4. On failure: `git reset --hard <start>`, `dvc checkout --force`, report the error.
5. On success: read `reports/train_metrics.json`, commit `dvc.lock` and `reports/`, report the new version.

**Deploy job** (`do_deploy`): `docker compose restart model-server`; poll `/ping` (up to 180 s);
`POST /model/reload`; `GET /ready` must show the target version and `model_server_in_sync`
not false.

The API key comes from the environment or `.env`; HTTP 401 is logged with a hint.

## 8. Dashboard (`frontend/`)

| File | Content |
|---|---|
| `app.py` | Navigation, sidebar system status (refreshed every 15 s) |
| `api_client.py` | `ApiClient`: one method per backend endpoint; `ApiError` with the error code |
| `ui.py` | Risk colours and labels, `show_api_error` (plain-language message per error code) |
| `views/predict.py` | Engine check: upload, fleet summary, ranked table, per-engine reasons |
| `views/pipeline.py` | DVC graph (Graphviz), stage metrics, MLflow runs, recent warnings |
| `views/monitoring.py` | Drift state and PSI chart, recent alerts, retraining jobs, manual retrain |
| `views/model.py` | Served model, sensors, champion vs challenger with checks, approve, roll back |
| `views/help.py` | In-app user manual and a CSV template |

The dashboard never imports `src/`; everything comes from the REST API.

## 9. Monitoring configuration (`monitoring/`)

| File | Content |
|---|---|
| `prometheus/prometheus.yml` | 15 s scrape; jobs: backend, mlflow, pushgateway (`honor_labels`), node, cadvisor (only `turbofan-*` containers kept), uptime (5 blackbox probes), prometheus, alertmanager |
| `prometheus/alert_rules.yml` | 11 rules: ServiceDown, ModelNotLoaded, ModelServerOutOfSync, HighErrorRate, HighPredictionLatency, DataDrift, PipelineStageFailed, RetrainingJobFailed, RetrainingJobStuck, ContainerMemoryHigh, DiskAlmostFull |
| `alertmanager/alertmanager.yml` | One route to the backend webhook, grouped by alert name, Bearer key from `/run/secrets/api_key`; model-server outage holds back HighErrorRate and HighPredictionLatency |
| `blackbox/blackbox.yml` | `http_2xx` module, 5 s timeout |
| `grafana/` | Data source and dashboard provisioned from files; dashboard JSON in `dashboards/turbofan.json` |
