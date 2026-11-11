# Test Plan: Turbofan Predictive Maintenance Platform

| | |
|---|---|
| Project | Predictive maintenance MLOps platform on NASA C-MAPSS (DA3408) |
| Document | Test plan (what is tested, how, and when it counts as passed) |
| Related | `docs/test_cases.md` (the cases), `docs/test_report.md` (the results) |

## 1. Purpose

The system predicts whether a turbofan engine will fail within the next 30 cycles, and
wraps that model in a full MLOps platform: data versioning, a Spark feature pipeline,
experiment tracking, a REST API, a dashboard, containers, monitoring with alerts, and a
retraining loop with human approval. This plan defines how each part is checked so that
the delivered system can be trusted by its two kinds of user:

- **Maintenance operator** (non-technical): uploads engine readings, reads the risk.
- **ML engineer / operator**: runs the pipeline, watches monitoring, approves models.

## 2. Scope

**In scope**

| Area | What is checked |
|---|---|
| Data ingestion | File reading, RUL labels, validation of raw files |
| Feature engineering | Rolling window features; Spark and pandas give the same numbers |
| Model evaluation | Metrics, decision threshold rule |
| Backend API | Every endpoint: answers, input validation, error format, API key |
| Dashboard | The helper logic behind the pages (messages, tables, sorting) |
| Monitoring | Prometheus metrics, PSI drift detection, alert path to the backend |
| Retraining loop | Job queue, drift trigger, approval gate, deploy, rollback |
| Whole system | All 11 containers together, as a user would use them |
| Non-functional | Response time, security settings, reproducibility, code quality |

**Out of scope**

- The accuracy of the NASA simulation itself (the dataset is taken as given).
- Load beyond a single laptop (no cloud, by project rule).
- Browser compatibility testing of the Streamlit dashboard (Chrome is used).
- Statistical optimisation of the model (the project goal is the platform, not the best score).

## 3. Test levels and approach

| Level | What it means here | How | Tool |
|---|---|---|---|
| **Unit** | One function or class alone, with small hand-made inputs whose answer is known | Automated | pytest |
| **Integration** | The FastAPI app with all its parts, but MLflow replaced by a fake (`FakeModelService`, `FakeRegistry`), so the API rules can be tested without servers | Automated | pytest + FastAPI `TestClient` |
| **System** | The real stack (`docker compose up`): real MLflow, model server, Prometheus, Alertmanager, worker | Manual, following written steps | Browser, terminal |
| **Non-functional** | Speed, security, reproducibility, code style | Script or command | `scripts/latency_test.py`, docker, git, flake8, black |

**Why fakes in the integration tests:** the API rules (for example "approval is refused if
FD001 recall drops by more than 0.05") must be tested in a few seconds, on any machine,
without starting MLflow. The fakes return fixed answers, so each test checks exactly one
rule. The real MLflow path is covered by the system tests.

**Why some modules have low unit-test coverage:** `train.py`, `model_service.py` and the
retraining worker need a live MLflow server, Spark or Docker. They are exercised end to end
by the system tests (a real training run, a real retrain job, a real deploy) instead.

**Test data**

- Unit tests build tiny synthetic tables in the test itself (for example 3 cycles of 1 engine).
- System tests use the files in `samples/`, made by `scripts/make_samples.py` from the
  official NASA test sets: normal FD001 engines, FD002 (new flight conditions), FD003
  (new fault), short histories and a file with a missing column.

## 4. Test environment

| Item | Value |
|---|---|
| Machine | Laptop, 12 CPU cores, 14 GiB RAM, Ubuntu 26.04.1 LTS |
| Python | 3.14.4 (project virtual environment `.venv`) |
| Containers | Docker 29.9.3, Docker Compose v5.7.0; images `da24b019/turbofan-backend:1.2`, `da24b019/turbofan-frontend:1.2` and pinned monitoring images |
| Data | DVC remote `localstore` (`/aiops-storage/dvc-storage`); versions fixed by `dvc.lock` |
| Model | Registered model `turbofan-failure-classifier`, champion version 8 |

## 5. Entry criteria (before testing starts)

1. The code is committed (`git status` is empty), so results belong to a known commit.
2. `dvc status` reports "Data and pipelines are up to date".
3. For system tests: `docker compose ps` shows all 11 containers up, the four application
   containers healthy, and the retraining worker running.

## 6. Exit criteria (when testing is finished)

1. Every automated test passes (79 of 79).
2. Every system and non-functional test case has been run and recorded with its result.
3. Every defect found is either fixed and re-tested, or recorded as a known limitation
   with a reason.
4. The acceptance criteria in section 7 are met, or the gap is explained in the test report.

## 7. Acceptance criteria

The system is accepted when all of these hold:

| ID | Criterion | Measured by |
|---|---|---|
| AC-01 | All automated tests pass | pytest: 79/79 |
| AC-02 | The prediction API answers one engine within 200 ms (95th percentile) | TC-P01 |
| AC-03 | A 10-engine CSV upload, including explanations, is answered within 500 ms (95th percentile) | TC-P02 |
| AC-04 | Bad input never crashes the API: it is rejected with HTTP 400/422 and a readable message in the common error format | TC-I03 |
| AC-05 | Protected endpoints refuse requests without the API key (HTTP 401) | TC-I04, TC-SEC01 |
| AC-06 | Normal data does not raise a drift alarm; data from new flight conditions (FD002) does | TC-S05, TC-S06 |
| AC-07 | A drift alarm reaches the backend as an alert without any person acting | TC-S07 |
| AC-08 | A drift alert starts a retraining job automatically; the new model is not used until a person approves it | TC-S09, TC-S10 |
| AC-09 | A model that is clearly worse on the original task (FD001) cannot be approved | TC-I08 |
| AC-10 | Rollback to an earlier model works and leaves the backend and model server in sync | TC-S12 |
| AC-11 | The project can be rebuilt from Git and the DVC remote alone, and its tests pass there | TC-S14 |
| AC-12 | Services are reachable only from the laptop (127.0.0.1), containers do not run as root, and no secrets are in Git | TC-SEC02 to TC-SEC04 |
| AC-13 | Model quality: cross-validated recall at least 0.90 (design target), FD001 test PR-AUC at least 0.90 | TC-S16 |
| AC-14 | Code passes flake8 with no findings and is black-formatted | TC-S15 |

## 8. How to run the tests

```bash
# Automated (unit + integration), with a JUnit report and coverage
python -m pytest -v --junitxml=reports/tests/junit.xml --cov=src --cov=frontend --cov-report=term

# Performance (stack must be running)
python -m scripts.latency_test

# Code quality
flake8 && black --check .
```

System test steps are listed case by case in `docs/test_cases.md`.

## 9. Defect handling

Each defect gets an ID (D-01, D-02, ...), the test that found it, the cause, the fix and
the re-test result. They are listed in `docs/test_report.md`, section 5. A defect is
closed only when the test that found it passes again.

## 10. Risks to testing

| Risk | Effect | How it is handled |
|---|---|---|
| Small test set: only 25 FD001 test engines are within 30 cycles of failure | One engine changes recall by 0.04, so recall is noisy | PR-AUC is reported next to recall; the approval gate allows a 0.05 change |
| Drift alarms depend on timing (scrape every 15 s, rule `for: 1m`) | Alert arrives 1 to 2 minutes after the cause | System test steps say how long to wait |
| System tests are manual | Could be skipped or done differently | Exact steps and expected results are written down; outputs saved as evidence |
| Laptop also runs other programs | Timing outliers | Performance criteria use the 95th percentile, not the maximum |
