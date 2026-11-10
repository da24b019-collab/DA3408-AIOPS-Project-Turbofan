# Test Cases

Each case has an ID, what it checks, how it is run, the expected result (the acceptance
criterion for that case) and the requirement it supports (AC-xx, see `docs/test_plan.md`).
Automated cases name the pytest functions that implement them. Results are in
`docs/test_report.md`.

ID prefixes: **U** unit, **I** integration (API with fakes), **S** system (full stack),
**P** performance, **SEC** security.

## 1. Unit tests (automated)

| ID | What is checked | Tests (file: functions) | Expected result |
|---|---|---|---|
| TC-U01 | Data ingestion: raw NASA files are read correctly, RUL labels are right, broken files are refused | `test_ingest.py`: `test_reader_handles_trailing_spaces`, `test_reader_rejects_wrong_column_count`, `test_train_rul_counts_down_to_zero`, `test_test_rul_adds_true_rul_at_end`, `test_test_labels_reject_wrong_rul_length`, `test_label_boundary_is_inclusive`, `test_validation_catches_missing_cycle`, `test_validation_catches_missing_values` (8) | RUL reaches 0 at the last cycle; RUL = 30 is labelled "fails soon" (inclusive); files with wrong columns, missing cycles or missing values raise an error |
| TC-U02 | Feature engineering: rolling mean and std over 10 cycles, per engine, same in Spark and pandas | `test_features.py`: `test_rolling_mean_by_hand`, `test_first_cycle_std_is_zero`, `test_engines_do_not_mix`, `test_unsorted_input_is_sorted`, `test_feature_names_order`, `test_missing_column_raises`, `test_spark_matches_pandas` (7) | Values equal a hand calculation; one engine's window never includes another engine; Spark (training) and pandas (serving) give the same numbers, so there is no training-serving skew |
| TC-U03 | Model evaluation and the threshold rule | `test_evaluate.py`: `test_metrics_by_hand`, `test_threshold_meets_target_recall`, `test_threshold_is_as_high_as_possible`, `test_always_safe_model_has_zero_recall` (4) | Metrics equal a hand calculation; the chosen threshold is the highest one that still reaches the target recall |
| TC-U04 | Drift detection (PSI) | `test_drift.py`: `test_baseline_bins_hold_ten_percent_each`, `test_repeated_values_do_not_create_duplicate_edges`, `test_psi_is_zero_for_identical_distributions`, `test_psi_grows_with_the_shift`, `test_empty_bins_do_not_give_infinity`, `test_monitor_states`, `test_new_baseline_empties_the_window`, `test_closest_of_several_baselines_is_used`, `test_old_single_baseline_format_still_works` (9) | PSI is 0 for identical data and grows with the shift; empty bins stay finite; states go no_baseline → collecting → ok → drift; with two baselines the closer one is used |
| TC-U05 | Pipeline keeps working when monitoring is down | `test_pipeline_metrics.py`: `test_push_failure_does_not_raise` (1) | Push to an unreachable Pushgateway returns False and raises nothing |
| TC-U06 | The retraining worker edits only the train section of params.yaml | `test_worker.py`: `test_only_train_subsets_change`, `test_missing_line_fails_clearly` (2) | `train.subsets` changes; `data.subsets` and comments stay the same; a missing line gives a clear error |

## 2. Integration tests: backend API (automated)

| ID | What is checked | Tests (file: functions) | Expected result | Req. |
|---|---|---|---|---|
| TC-I01 | Health and readiness | `test_api.py`: `test_health`, `test_ready_when_model_loaded`, `test_not_ready_without_model`, `test_not_ready_when_model_server_out_of_sync` (4) | `/health` 200 always; `/ready` 200 only with a model loaded and the model server in sync, else 503 | |
| TC-I02 | Prediction, normal use | `test_predict_json_two_engines`, `test_predict_csv_upload`, `test_risk_level_boundaries`, `test_summary_lists_reasons_naturally`, `test_short_history_gives_warning` (5) | One result per engine, with probability, risk level and reasons; engines with fewer than 10 cycles get a warning, not an error | |
| TC-I03 | Input validation | `test_missing_column_is_422`, `test_duplicate_cycle_is_422`, `test_non_csv_upload_is_400`, `test_too_many_rows`, `test_wrong_json_shape_uses_same_error_format`, `test_column_names_are_case_insensitive` (6) | Bad input gives 400 or 422 with `{"error", "detail"}` naming the problem; the API never crashes | AC-04 |
| TC-I04 | Authentication | `test_api_key_required_when_set`, `test_alert_webhook_accepts_bearer_token_and_stores_alert` (2) | Without the key: 401. Key accepted as `X-API-Key` or `Authorization: Bearer` | AC-05 |
| TC-I05 | Prometheus metrics | `test_metrics_page_is_open_and_counts_predictions`, `test_unknown_paths_are_grouped_in_metrics`, `test_validation_errors_are_counted` (3) | `/metrics` open; counters rise after requests; unknown URLs grouped as `other` | |
| TC-I06 | Drift endpoints | `test_drift_without_baseline`, `test_drift_collecting_then_ok_then_drift` (2) | State follows the readings received; reset empties the window | AC-06 |
| TC-I07 | Pipeline information | `test_pipeline_info.py`: `test_stages_and_arrows`, `test_stage_metrics_are_extracted`, `test_log_events_keep_only_warnings_and_errors`, `test_missing_dvc_yaml`, `test_runs_endpoint_reports_mlflow_down` (5) | DAG read from dvc.yaml; MLflow being down gives 503, not a crash | |
| TC-I08 | Retraining loop rules | `test_retraining.py`: `test_job_lifecycle`, `test_drift_alert_queues_one_retrain_job`, `test_cooldown_after_a_finished_job`, `test_resolved_alert_does_not_retrain`, `test_worker_claims_and_finishes`, `test_approve_moves_alias_and_queues_deploy`, `test_approve_refused_when_fd001_gets_worse`, `test_only_one_deploy_at_a_time`, `test_rollback` (9) | One drift alert = one job (no duplicates, 30-min cooldown); approval refused (409) when FD001 recall drops by more than 0.05, and nothing changes; one deploy at a time; unknown version 404 | AC-08, AC-09 |
| TC-I09 | Dashboard helper logic | `test_frontend.py`: 12 tests (risk wording never claims certainty, riskiest engines first, friendly error when backend is down, drift messages, PSI table, comparison table, jobs table, pipeline diagram) | Every page helper returns the expected text or table | |

Total automated: 8 + 7 + 4 + 9 + 1 + 2 + 22 + 5 + 9 + 12 = **79 tests**.

## 3. System tests (manual, full stack)

Precondition for all: entry criteria of the test plan met (stack up, worker running).

| ID | What is checked | Steps | Expected result | Req. |
|---|---|---|---|---|
| TC-S01 | The whole system starts | `docker compose up -d`; `docker compose ps` | 11 containers up; mlflow, model-server, backend, frontend healthy | |
| TC-S02 | The pipeline reproduces | `dvc repro` | ingest → features → train run; MLflow run logged with git commit and DVC data version; new model version registered as challenger; metrics pushed to the Pushgateway | |
| TC-S03 | An operator checks engines | Dashboard, Engine check: upload `samples/fd001_mixed_engines.csv` | A table of 10 engines, riskiest first, each with risk level and the sensors behind it | |
| TC-S04 | Monitoring collects from everything | Open `localhost:9090/targets` | All 12 targets up; all 5 uptime probes 1 | |
| TC-S05 | No false drift alarm on normal data | Monitoring: empty the drift window; upload `samples/drift_fd001_normal.csv` | "No drift", highest PSI below 1.0 | AC-06 |
| TC-S06 | Drift is detected | Empty the window; upload `samples/drift_fd002_conditions.csv` | "Data drift detected", PSI above 1.0 | AC-06 |
| TC-S07 | The alert travels by itself | After TC-S06, wait 2 min; then empty the window and wait 2 min | DataDrift firing in Prometheus, Alertmanager and the dashboard's Recent alerts; then a "resolved" row | AC-07 |
| TC-S08 | A real resource problem raises an alert | Observed during operation | ContainerMemoryHigh fires when a container uses over 90% of its memory limit, and resolves when fixed | |
| TC-S09 | Drift starts retraining automatically | Continue from TC-S06 with the worker running | A retrain job appears in the same second as the alert; the worker trains a challenger and the job ends "succeeded" | AC-08 |
| TC-S10 | Human approval | Model page: read the comparison, tick the box, Approve | Approve is disabled until the box is ticked; after approval a deploy job restarts the model server and ends "in sync" | AC-08 |
| TC-S11 | The loop closes | Empty the window; upload the FD002 file again | "No drift": the new model was trained on FD002, so FD002 is known data | AC-06 |
| TC-S12 | Rollback | Model page, Roll back to the previous version; then approve again | Deploy job ends "in sync" with the old version; the newer one is offered again for approval | AC-10 |
| TC-S13 | A failed training run leaves no damage | Run a retrain job where `dvc repro` fails | Job "failed" with the error; Git back at the starting commit; `git status` empty; outputs restored | |
| TC-S14 | Reproducible from scratch | Clone the repo to a new folder; new venv from `requirements.txt`; `dvc pull`; `dvc status`; pytest | All steps work; "up to date"; 79 passed | AC-11 |
| TC-S15 | Code quality | `flake8 && black --check .` | No flake8 findings; black: all files unchanged | AC-14 |
| TC-S16 | Model quality | Read the champion's run in MLflow and the Model page comparison | CV recall at least 0.90; FD001 test PR-AUC at least 0.90 | AC-13 |

## 4. Performance tests

| ID | What is checked | Steps | Expected result | Req. |
|---|---|---|---|---|
| TC-P01 | Response time, one engine | `python -m scripts.latency_test` (100 JSON requests of one engine, 31 cycles) | 95th percentile below 200 ms | AC-02 |
| TC-P02 | Response time, batch upload | Same script (20 CSV uploads of 10 engines, 1,433 rows, with SHAP reasons) | 95th percentile below 500 ms | AC-03 |

## 5. Security tests

| ID | What is checked | Steps | Expected result | Req. |
|---|---|---|---|---|
| TC-SEC01 | API refuses unauthenticated requests | `curl localhost:8000/model/info` without a key | HTTP 401 | AC-05 |
| TC-SEC02 | Services only reachable from the laptop | `docker compose ps --format "{{.Name}} {{.Ports}}"` | Every published port starts with `127.0.0.1:` | AC-12 |
| TC-SEC03 | Containers do not run as root | `docker exec <container> id -u` for the four application containers | uid 1000, not 0 | AC-12 |
| TC-SEC04 | No secrets in Git | `git ls-files \| grep .env`; fresh clone has no `.env` | No `.env` in the repository | AC-12 |
