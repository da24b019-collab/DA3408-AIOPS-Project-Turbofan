# Test Report

| | |
|---|---|
| Project | Turbofan Predictive Maintenance Platform (DA3408) |
| Test period | 2026-11-07 to 2026-11-10 |
| Environment | Laptop, Ubuntu 26.04.1 LTS, 12 cores, 14 GiB RAM; Python 3.14.4; Docker 29.9.3, Compose v5.7.0 |
| Software under test | Commit after "requirements: include the dashboard's libraries" (8db0248) plus Task 10 files; images `turbofan-backend:1.2`, `turbofan-frontend:1.2`; champion model version 8 |
| Plan and cases | `docs/test_plan.md`, `docs/test_cases.md` |
| Raw evidence | `reports/tests/junit.xml`, `reports/tests/pytest_output.txt`, `reports/tests/latency.txt`, screenshots in `docs/screenshots/` |

## 1. Summary

| Kind | Cases | Passed | Failed | Notes |
|---|---|---|---|---|
| Automated tests (unit + integration) | 79 tests in 15 cases | **79** | 0 | 15.5 s |
| System tests | 16 | **16** | 0 | TC-S14 failed on its first run (defect D-09), passed after the fix; TC-S13 run in the development environment |
| Performance tests | 2 | **2** | 0 | |
| Security tests | 4 | **4** | 0 | |
| **Acceptance criteria** | 14 | **14 met** | 0 | AC-13: FD001 test recall reported separately as a known limitation (section 6) |

**Conclusion:** all exit criteria of the test plan are met. Every defect found was fixed and
re-tested (section 5). The system is accepted, with the limitations listed in section 6.

## 2. Automated tests

Command: `python -m pytest -v --junitxml=reports/tests/junit.xml --cov=src --cov=frontend --cov-report=term`
Result: **79 passed, 0 failed, 1 warning** (a deprecation notice from inside the Starlette
test client library, not from project code).

| Case | Area | Tests | Result |
|---|---|---|---|
| TC-U01 | Data ingestion | 8 | Pass |
| TC-U02 | Feature engineering, Spark-pandas parity | 7 | Pass |
| TC-U03 | Evaluation, threshold rule | 4 | Pass |
| TC-U04 | Drift detection (PSI) | 9 | Pass |
| TC-U05 | Pipeline survives monitoring outage | 1 | Pass |
| TC-U06 | Worker edits params.yaml safely | 2 | Pass |
| TC-I01 | Health and readiness | 4 | Pass |
| TC-I02 | Prediction, normal use | 5 | Pass |
| TC-I03 | Input validation | 6 | Pass |
| TC-I04 | Authentication | 2 | Pass |
| TC-I05 | Prometheus metrics | 3 | Pass |
| TC-I06 | Drift endpoints | 2 | Pass |
| TC-I07 | Pipeline information | 5 | Pass |
| TC-I08 | Retraining loop rules | 9 | Pass |
| TC-I09 | Dashboard helper logic | 12 | Pass |

### Code coverage

Overall statement coverage: **52%** (2,076 statements, 994 not run by the automated tests).

| Group | Modules | Coverage |
|---|---|---|
| Core logic, fully unit-tested | `drift.py`, `jobs.py`, `metrics.py`, `schemas.py`, `settings.py`, `feature_logic.py`, `config.py` | 100% |
| | `prediction.py` 97%, `logger.py` 95%, `main.py` (API) 91%, `pipeline_metrics.py` 91%, `engine_data.py` 84% | 84 to 97% |
| Needs a live MLflow server, Spark or Docker | `train.py` 0%, `promote.py` 0%, `model_service.py` 34%, `build_features.py` 33%, `worker.py` 27%, `registry.py` 39%, `compare.py` 42% | 0 to 42% |
| Dashboard page layout (Streamlit drawing code) | `app.py` 0%, `help.py` 0%, page files 21 to 33% | 0 to 33% |

The low numbers are concentrated in code that talks to external services. That code is
not untested: it ran for real in the system tests. `train.py` and `worker.py` ran in
TC-S02 and TC-S09, `model_service.py` and `registry.py` in TC-S03, TC-S10 and TC-S12,
`build_features.py` in TC-S02, and the dashboard pages in every manual case. A unit test
of them would mostly test the fakes.

## 3. System tests

| Case | Date | Result | Evidence (actual result) |
|---|---|---|---|
| TC-S01 Stack starts | 2026-11-10 | Pass | `up 11/11`; mlflow, model-server, backend healthy |
| TC-S02 Pipeline reproduces | 2026-11-07 | Pass | `dvc repro`: features 265,256 rows in 34.1 s (7,780 rows/s), train registered version 7 as challenger; both stages pushed metrics to the Pushgateway |
| TC-S03 Operator checks engines | 2026-11-07 | Pass | Upload of `drift_fd001_normal.csv`: 26 engines predicted; sensor rows received 3,062 (Grafana) |
| TC-S04 Monitoring collects from everything | 2026-11-07 | Pass | All uptime probes UP in Grafana; all scrape targets up |
| TC-S05 No false drift alarm | 2026-11-07 16:12 | Pass | "No drift", highest PSI 0.85 (s11), 3,062 readings |
| TC-S06 Drift detected | 2026-11-07 16:15 | Pass | "Data drift detected", highest PSI 4.58 (s17), 13 of 14 sensors above 1.0 |
| TC-S07 Alert travels by itself | 2026-11-07 | Pass | DataDrift firing received 16:16:58, resolved 16:17:58 after the window was emptied |
| TC-S08 Real resource alert | 2026-11-07 | Pass | ContainerMemoryHigh for Grafana fired 16:09 and 16:17; after the fix (D-04) resolved 16:19:31 and did not return |
| TC-S09 Drift starts retraining | 2026-11-07 | Pass | Alert 16:55:13, retrain job 1 created the same second; worker trained version 8 on FD001+FD002 in 101 s; job "succeeded" |
| TC-S10 Human approval | 2026-11-07 | Pass | Approve disabled until the box was ticked; approved 16:58:22; deploy job 2 "in sync" after 24 s |
| TC-S11 Loop closes | 2026-11-07 | Pass | Same FD002 file after deployment: "No drift" |
| TC-S12 Rollback | 2026-11-07 | Pass | Rollback to version 7 (job 3, 24 s, in sync); version 8 approved again (job 4, 24 s, in sync) |
| TC-S13 Failed training leaves no damage | development env. | Pass | Retrain job with a failing `dvc repro`: job "failed" with the error; `git status` empty, HEAD back at the start commit, params.yaml unchanged, reports restored |
| TC-S14 Reproducible from scratch | 2026-11-10 | Pass (2nd run) | Run 1 **failed**: tests could not import `streamlit` (D-09). After the fix: `dvc pull` 47 files, "up to date", 79 passed, style OK |
| TC-S15 Code quality | 2026-11-07 | Pass | flake8: no findings; black: 59 files unchanged |
| TC-S16 Model quality | 2026-11-07 | Pass | Version 8: threshold chosen for CV recall 0.90; FD001 test PR-AUC 0.943; FD002 test recall 0.967 |

## 4. Performance and security

| Case | Result | Measured | Target |
|---|---|---|---|
| TC-P01 One engine (JSON), 100 requests | **Pass** | median 35.7 ms, **p95 49.0 ms**, max 1,482 ms | p95 < 200 ms |
| TC-P02 10 engines (CSV + SHAP), 20 requests | **Pass** | median 75.0 ms, **p95 96.3 ms**, max 99.2 ms | p95 < 500 ms |
| TC-SEC01 No API key | **Pass** | HTTP 401 | 401 |
| TC-SEC02 Published ports | **Pass** | all 9 published ports on `127.0.0.1`; blackbox and node exporters not published at all | 127.0.0.1 only |
| TC-SEC03 Container user | **Pass** | backend, frontend, mlflow, model-server: uid 1000 | not 0 (root) |
| TC-SEC04 Secrets in Git | **Pass** | no `.env` in Git; fresh clone has no `.env` | none |

The single 1,482 ms request in TC-P01 is 1 of 100. The backend container had restarted a
few seconds earlier, and the batch test that followed had no such spike, so it is a
one-off delay, not a trend. This is why the criterion uses the 95th percentile.

## 5. Defects found and fixed

| ID | Found by | Problem | Cause | Fix | Re-test |
|---|---|---|---|---|---|
| D-01 | Experiment (model selection) | Random train/validation split gave PR-AUC 0.982 instead of 0.949 | Rows of the same engine in both training and validation (leakage) | GroupKFold by engine | Pass |
| D-02 | Drift study (Task 8) | 49% false drift alarms on normal data at 1,000 rows | First check sampled random single rows; real uploads are whole engines | Window 5,000 rows, judged from 3,000 (0 to 1% false alarms) | Pass (TC-S05) |
| D-03 | TC-S04 (Grafana check) | "Model served" showed v6 and v7; "Error rate" showed No data | Range query kept the old version; no 5xx series when there are no errors | Instant query; `or vector(0)` | Pass |
| D-04 | TC-S08 (live alert) | Grafana above 90% of its memory limit | 512 MB limit too small for Grafana 13 | Limit 768 MB | Pass (alert resolved) |
| D-05 | Development (Task 9) | Retraining worker code missing from Git | `.gitignore` rule `retraining/` also matched `src/retraining/` | Rule anchored: `/retraining/` | Pass |
| D-06 | Development (Task 9) | Tests wrote a jobs file into the real project | Test settings used the real project folder | Tests use a temporary folder | Pass |
| D-07 | Measurement (Task 9) | Model trained on FD001+FD002 flagged pure FD001 data as drift (PSI 4.9) | One baseline of mixed data | One baseline per training dataset, closest one used | Pass (TC-S11) |
| D-08 | TC-S15 (flake8) | Drift module comment said 500 rows / 1,000-row window | Comment not updated after D-02 | Comment corrected | Pass |
| D-09 | TC-S14 (fresh clone) | Tests could not import `streamlit` | Dashboard libraries listed only in `frontend/requirements.txt`; worked locally because of a manual install | Main requirements include the frontend file (`-r`) | Pass (2nd run) |
| D-10 | Dashboard review (Task 9) | Recent alerts table did not show what the backend did about an alert | Column left out of the table | `action` column added | Pass |

## 6. Known limitations

1. **FD001 test recall is 0.76**, below the cross-validated 0.90. The FD001 test set has
   only 25 engines within 30 cycles of failure, so each engine moves recall by 0.04; the
   difference is 4 engines. PR-AUC (0.943, which does not depend on one threshold) is the
   steadier measure and meets AC-13.
2. **FD003 (second fault mode) is detected as drift in about 9 of 10 windows**, not always,
   and neither model predicts FD003 failures well (recall 0.25 to 0.30), because no model
   has been trained on that fault.
3. **The retraining worker must be running.** If it is not, jobs wait; the
   RetrainingJobStuck alert fires after 30 minutes.
4. **Job counters in Prometheus reset when the backend restarts** (the jobs file does not).
5. **TC-S13 was run in the development environment**, not on the final laptop, to avoid
   breaking the working project on purpose.
6. **Dashboard drawing code is tested manually**, not by automated tests (section 2).
