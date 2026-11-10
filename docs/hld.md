# High-Level Design: Turbofan Predictive Maintenance Platform

| | |
|---|---|
| Course | DA5402 (MLOps / AIOps) |
| Related | Architecture diagrams: `docs/diagrams/`; low-level design: `docs/lld.md` and `docs/lld_api.md`; tests: `docs/test_plan.md` |

## 1. Problem and goal

Aircraft engines wear out. If an engine fails in service it is dangerous and costly; if it is
serviced too early, money is wasted. The goal is to tell a maintenance team, from an engine's
recent sensor readings, **whether it is likely to fail within the next 30 operating cycles**,
so it can be serviced in time.

The data is the NASA **C-MAPSS** simulation: four datasets (FD001 to FD004) of engines run
until failure, each cycle with 3 operating settings and 21 sensors.

| Dataset | Flight conditions | Fault types | Role in this project |
|---|---|---|---|
| FD001 | 1 | 1 (HPC degradation) | Original training data |
| FD002 | 6 | 1 | "New conditions": drift demonstration and retraining data |
| FD003 | 1 | 2 | "New fault": drift test |
| FD004 | 6 | 2 | Evaluation only |

The project's emphasis is the **platform around the model** (MLOps), not the most accurate
model: versioned data, a reproducible pipeline, tracked experiments, a served model with a
user interface, containers, monitoring, and a controlled way to replace the model.

## 2. Requirements

**Functional**

| ID | Requirement | Where it is met |
|---|---|---|
| F1 | An operator uploads engine readings (CSV) and sees each engine's failure risk, riskiest first, with the sensors that drove it | Dashboard "Engine check" page, `POST /predict/csv` |
| F2 | The operator can understand the result without ML knowledge | Risk levels high / medium / low, one-sentence summary, in-app Help page |
| F3 | Bad input is rejected with a clear message, never a crash | Validation in `engine_data.py`, uniform error format |
| F4 | Every model can be traced to its code, parameters and data | MLflow run tags: git commit, dirty flag, DVC data hash |
| F5 | The pipeline can be rerun with one command and gives the same result | DVC `dvc repro`, fixed random seed |
| F6 | The system watches itself and alerts on problems, including data drift | Prometheus, Alertmanager, PSI drift check |
| F7 | Drift starts retraining automatically; a person approves before a new model is used | Job queue, retraining worker, approval gate, Model page |
| F8 | A bad model can be rolled back quickly | Rollback on the Model page, deploy job |
| F9 | Engineers can see the pipeline and its runs from the dashboard | "Pipeline" page, `GET /pipeline/status`, `/pipeline/runs` |

**Non-functional**

| ID | Requirement | Target | Achieved (test report) |
|---|---|---|---|
| N1 | Response time, one engine | p95 < 200 ms | 49 ms |
| N2 | Response time, 10-engine upload with explanations | p95 < 500 ms | 96 ms |
| N3 | Runs on one laptop, no cloud | 14 GiB RAM, 12 cores | All 11 containers within memory limits |
| N4 | Reproducible by another person | From Git + DVC remote only | Fresh-clone test passed |
| N5 | Secure by default | Local-only ports, API key, no root, no secrets in Git | All four security tests passed |
| N6 | Maintainable | Tests, style checks, documented design | 79 tests, flake8 clean, black-formatted |

## 3. Architecture overview

![System architecture](diagrams/architecture.png)

The system has three parts (source of the diagram: `diagrams/architecture.dot`):

1. **Application containers**: the dashboard (`frontend`), the REST API (`backend`), the
   model server (`model-server`) and the MLflow tracking server and registry (`mlflow`).
2. **Monitoring containers**: Prometheus and its exporters, Alertmanager, Pushgateway and Grafana.
3. **The laptop itself**: the DVC pipeline (with Spark), the retraining worker, the Git
   repository and the DVC remote.

The model's life cycle, including the retraining loop:

![ML lifecycle](diagrams/ml_lifecycle.png)

### Main flows

| Flow | Path |
|---|---|
| Prediction | Browser → frontend → backend (validate, compute 42 features, keep last cycle) → model server (probability) → backend (risk level, SHAP reasons) → frontend |
| Training | `dvc repro`: ingest → features (Spark) → train → MLflow (run + new version as challenger) → Pushgateway (stage metrics) |
| Monitoring | Prometheus scrapes every 15 s → rules → Alertmanager → backend webhook; Grafana reads Prometheus |
| Retraining | DataDrift alert → backend queues job → worker: commit params, `dvc repro` → challenger → human approval → deploy job → model server restart → in-sync check |

## 4. Design decisions

Each choice below lists the option taken, the alternative, and why.

### 4.1 Data and pipeline

| Decision | Alternative | Reason |
|---|---|---|
| **DVC** for data and model versions, with a local folder as remote | Git LFS for everything; no versioning | DVC knows the pipeline: it reruns only the stages whose inputs changed and records exact data hashes in `dvc.lock`. Local remote because the project must not use cloud. |
| **Git LFS** for binary files that belong with the code (figures, PDFs, screenshots) | Plain Git | Git stores a full copy of each binary version; LFS stores a small pointer. These files are not pipeline outputs, so DVC is not the right tool for them. |
| Label: **fails within 30 cycles** (classification) | Predict the exact remaining life (regression) | A maintenance decision is yes or no ("service it now?"). A probability with a threshold is easier for operators to act on. |
| **14 of 21 sensors**, window **10 cycles** | All sensors | EDA showed 6 sensors are constant in FD001 and one is pure noise; they add nothing but risk. |
| Features: raw value, 10-cycle mean, 10-cycle std per sensor (42) | Deep learning on the raw sequence | Simple, explainable features; the trend (mean) and instability (std) are what degradation looks like. |
| **Apache Spark** for feature engineering | pandas only | The project shows a scalable pipeline. Measured honestly: on C-MAPSS (265k rows) pandas is faster; Spark wins above about 800k rows and is 1.8x faster at 3.2M rows on 12 cores. The serving path uses pandas (no Spark start-up delay), and a test proves both give identical numbers, which prevents training-serving skew. |

### 4.2 Model and experiments

| Decision | Alternative | Reason |
|---|---|---|
| **XGBoost** | Random forest, logistic regression | Compared in `experiments/model_cost_benchmark.py`: random forest gave similar quality but a model 17x larger and 23x slower to predict; logistic regression cannot learn the non-linear patterns of mixed flight conditions. |
| **GroupKFold by engine** | Random row split | A random split put rows of the same engine in training and validation and inflated PR-AUC from 0.949 to 0.982 (leakage). |
| Threshold: highest that still gives **recall ≥ 0.90** in cross-validation | Fixed 0.5 | Missing a failing engine is worse than an extra inspection, so recall is the priority. |
| **MLflow** tracking, registry with aliases **champion / challenger** | Model files in folders | Every run stores parameters, metrics, plots, the Git commit and the DVC data hash. Aliases let the serving side ask for "the champion" instead of a fixed version. |
| Model served by **`mlflow models serve`** in its own container | Model inside the API | Uses MLflow's standard REST serving; the model can be replaced without rebuilding the API image. The backend keeps a local copy only for SHAP explanations and checks that both copies give the same answers (sync check). |

### 4.3 Application

| Decision | Alternative | Reason |
|---|---|---|
| **FastAPI** backend, **Streamlit** frontend as separate programs talking only over REST | One combined app | The frontend has no access to models or data; any client (curl, a future mobile app) gets the same API. The API contract is documented (`lld_api.md`) and versioned with the code. |
| API key on every endpoint except `/health`, `/ready`, `/metrics` | No authentication | Predictions and model control should not be open, even on a laptop. Health and metrics hold no data and must be open for probes. |
| Uniform error format `{"error", "detail"}`, validation before any model call | Framework defaults | The dashboard can show one kind of message for every failure. |

### 4.4 Deployment

| Decision | Alternative | Reason |
|---|---|---|
| **Docker Compose**, 11 containers, images on **Docker Hub** (`da24b019/turbofan-*`) | Run everything by hand | One command starts the whole system the same way everywhere. Health checks start containers in order. |
| Ports published on **127.0.0.1** only; containers run as the user's uid; memory limit per container | Defaults (all interfaces, root) | Nothing is reachable from the network; a broken container cannot take all the memory. The Grafana limit was raised after the monitoring itself showed it was too low. |
| MLflow job execution disabled | Default | Measured 2,672 MB of memory with it on versus 303 MB off; the feature is not used. |

### 4.5 Monitoring and drift

| Decision | Alternative | Reason |
|---|---|---|
| **Prometheus + Grafana + Alertmanager**, exporters for host, containers and uptime | Logs only | Numbers over time, alert rules and dashboards; the industry-standard open-source stack. |
| **Pushgateway** for pipeline stages | None | A pipeline stage runs for a minute and exits, so Prometheus cannot scrape it; it pushes its result instead. |
| Drift with **PSI**, threshold **1.0**, judged from **3,000** rows, window **5,000** | Textbook threshold 0.25; statistical tests | Measured on whole-engine windows: normal FD001 data never scores below 0.18, so 0.25 would always alarm; 1,000-row windows gave 49% false alarms. The chosen values give 0 to 1% false alarms and always detect FD002. Statistical tests flag even harmless differences at these sample sizes. |
| **One drift baseline per training dataset**, saved in the model's own MLflow run | One mixed baseline | A mixed FD001+FD002 baseline called pure FD001 data drifted (PSI 4.9). |

### 4.6 Retraining loop

| Decision | Alternative | Reason |
|---|---|---|
| Backend only **queues** jobs; a **worker on the laptop** does them | Backend trains and restarts containers | Training needs Spark, DVC and Git; restarting needs Docker. Giving the API container control of Docker would let an attacker of the API control every container. The worker pulls work, like a CI runner. |
| One open retrain job at a time, 30-minute cooldown for alert-triggered jobs | Retrain on every alert | A firing alert repeats; without limits it would start training after training. |
| Worker refuses to run with uncommitted changes, commits the params change and results | Train from whatever is on disk | Each model version points to an exact commit. |
| **Approval gate** (FD001 recall and PR-AUC must not drop by more than 0.05), then a **human** approves | Automatic promotion | The retrained model went from 0.000 to 0.967 recall on FD002 but from 0.80 to 0.76 on FD001: a trade-off a person should accept knowingly. |
| Rollback without the gate | Gate everything | In an emergency speed matters; the older version was approved when first used. |

## 5. Programming style

The code mixes two styles on purpose:

- **Functions** for everything that turns inputs into outputs: each pipeline stage, the
  feature logic, metrics, PSI. Functions with no hidden state are easy to test with small
  hand-made inputs (most of the 79 tests do exactly that).
- **Classes only where state must be kept between calls**: `Settings` (configuration),
  `ModelService` (the loaded model and explainer, with a lock for safe swapping),
  `DriftMonitor` (the window of recent readings), `JobStore` (the job queue file) and
  `Registry` (MLflow access, replaceable by a fake in tests).

All tunable values live in **`params.yaml`** (read by DVC, training and the API), and all
secrets in **`.env`** (never in Git).

## 6. Security summary

| Measure | Purpose |
|---|---|
| Ports bound to 127.0.0.1 | Not reachable from other machines |
| API key (`X-API-Key` or `Authorization: Bearer`) | Only the dashboard, the worker and Alertmanager can call the API |
| Secrets only in `.env`, passed to Alertmanager as a Compose secret file | No keys in Git or in config files |
| Containers run as uid 1000, memory limits | Limits damage from a broken or compromised container |
| No Docker access for the API | A compromised API cannot control containers |
| Grafana admin password from `.env`, sign-up disabled | No default login |

## 7. Known limitations

1. FD001 test recall is 0.76 with only 25 failing test engines (each engine is 0.04).
2. A new fault mode (FD003) is detected as drift about 9 times in 10 and is not learned.
3. In a real fleet, retraining would use newly collected, labelled failures; here the FD002
   training set plays that part.
4. The worker must be running for jobs to be done (an alert fires after 30 minutes if not).
5. Single machine: no high availability; MLflow uses SQLite.
