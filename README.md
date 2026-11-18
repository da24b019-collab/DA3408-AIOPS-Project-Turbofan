# Turbofan Predictive Maintenance (DA3408 MLOps project)

Predicts whether an aircraft engine will **fail within the next 30 operating cycles**, from
its recent sensor readings (NASA C-MAPSS data), and runs that model on a complete MLOps
platform: versioned data, a Spark pipeline, tracked experiments, a REST API with a
dashboard, containers, monitoring with alerts, and a retraining loop with human approval.
Everything runs on one laptop; no cloud services are used.

![System architecture](docs/diagrams/architecture.png)

**Demonstration Video:** https://drive.google.com/drive/folders/1Q7HTqtdjsXxgfeaqOTC1jeDVuBN4oCXJ?usp=drive_link


## Tools and where they are used

| Tool | Used for | Where |
|---|---|---|
| **Git** + **Git LFS** | Code, parameters, pipeline lock file; binary files (figures, PDFs) via LFS | whole repository, `.gitattributes` |
| **DVC** | Pipeline (ingest → features → train) and versions of data and models; local remote | `dvc.yaml`, `dvc.lock`, `/aiops-storage/dvc-storage` |
| **Apache Spark** | Rolling-window feature engineering | `src/features/build_features.py` |
| **MLflow** | Experiment tracking, model registry (champion / challenger), model serving, MLproject | `src/models/`, `MLproject`, `mlflow` and `model-server` containers |
| **XGBoost**, **SHAP** | Classifier; per-engine explanations | `src/models/train.py`, `src/api/model_service.py` |
| **FastAPI** | REST backend | `src/api/` |
| **Streamlit** | Dashboard for operators | `frontend/` |
| **Docker**, **Docker Compose**, **Docker Hub** | 11 containers; images `da24b019/turbofan-backend`, `da24b019/turbofan-frontend` | `docker-compose.yml`, `backend/`, `frontend/` |
| **Prometheus**, **Alertmanager**, **Pushgateway**, exporters | Metrics, 11 alert rules, pipeline metrics, uptime / host / container metrics | `monitoring/` |
| **Grafana** | Dashboard of the whole system, provisioned from files | `monitoring/grafana/` |
| **pytest**, **flake8**, **black** | 79 tests, code checks, formatting | `tests/`, `.flake8`, `pyproject.toml` |

## Results in brief

| | |
|---|---|
| Model (retrained on FD001 + FD002) | FD001 test PR-AUC 0.943; FD002 test recall 0.967 (the FD001-only model: 0.000) |
| Response time | one engine p95 49 ms; 10-engine upload with explanations p95 96 ms |
| Drift detection | normal data 0 to 1% false alarms; new flight conditions always detected |
| Retraining loop | drift alert → retrained challenger in under 2 minutes; deploy after approval 24 s |
| Tests | 79 automated (all pass), 16 system, 2 performance, 4 security; see `docs/test_report.md` |

## Setup on a new machine

**Prerequisites**: Ubuntu (or another Linux), Python 3.14, Java 17 or newer (for Spark),
Docker with Compose v2, Git with Git LFS, and access to the DVC remote folder
(`/aiops-storage/dvc-storage`, or change it with `dvc remote modify localstore url <path>`).

```bash
# 1. Code (Git LFS brings the figures and PDFs)
git clone https://github.com/da24b019-collab/DA3408-AIOPS-Project-Turbofan.git turbofan-pdm && cd turbofan-pdm
git lfs pull

# 2. Python environment
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Data and model files from the DVC remote
dvc pull
dvc status                      # "Data and pipelines are up to date."

# 4. Settings and secrets (never committed)
cp .env.example .env
#   HOST_UID / HOST_GID: output of `id -u` / `id -g`
#   API_KEY:             openssl rand -hex 24
#   GRAFANA_ADMIN_PASSWORD: openssl rand -hex 12
#   DOCKER_DATA_DIR:     docker info --format '{{ .DockerRootDir }}'
#   DOCKERHUB_USER=da24b019, IMAGE_TAG=1.2  (or build your own images)

# 5. Start the system
docker compose pull backend frontend    # or: docker compose build
docker compose up -d mlflow

# 6. First model (the MLflow registry starts empty on a new machine)
dvc repro -f -s train                   # trains and registers version 1 as challenger
python -m src.models.promote --approve  # make it champion
docker compose up -d                    # start everything else
docker compose ps                       # 4 application containers "healthy"

# 7. Retraining worker (second terminal, keep it running)
source .venv/bin/activate && python -m src.retraining.worker

# 8. Check
python -m pytest -q                     # 79 passed
```

Then open the dashboard at http://localhost:8501. All addresses and logins are in
`docs/user_manual.md`, section B1.

## Everyday commands

```bash
dvc repro                         # rerun the pipeline stages whose inputs changed
dvc push                          # store new data / model versions in the remote
python -m src.models.promote      # compare champion and challenger
python -m scripts.latency_test    # performance test against the running stack
flake8 && black --check .         # code style
docker compose logs -f backend    # follow the API's log
```

## Project structure

```
src/
  data/ingest.py                stage 1: read, validate, label
  features/                     stage 2: Spark features (+ the same logic in pandas for serving)
  models/                       stage 3: train, evaluate, compare, promote, tracking helpers
  monitoring/                   PSI drift detection, Pushgateway metrics
  retraining/worker.py          does retrain and deploy jobs
  api/                          FastAPI backend
frontend/                       Streamlit dashboard (talks to the backend only over REST)
backend/Dockerfile              image for backend, mlflow and model-server
monitoring/                     Prometheus, Alertmanager, blackbox, Grafana configuration
tests/                          79 automated tests
scripts/                        sample files, performance test
experiments/                    benchmarks and studies behind design decisions
notebooks/01_eda.ipynb          exploratory data analysis
samples/                        CSV files for trying the dashboard and the drift demo
docs/                           design documents, test documents, user manual
params.yaml                     every tunable value
dvc.yaml / dvc.lock             pipeline definition / exact versions of a run
docker-compose.yml              the whole system
```

## Documents

| Document | Content |
|---|---|
| `docs/diagrams/` | Architecture and ML lifecycle diagrams (Graphviz source + PNG + SVG) |
| `docs/hld.md` | High-level design: requirements, architecture, design decisions with reasons |
| `docs/lld.md` | Low-level design of the pipeline, model, monitoring and retraining modules |
| `docs/lld_api.md` | Low-level design of the REST API: every endpoint, request, response and error |
| `docs/test_plan.md`, `docs/test_cases.md`, `docs/test_report.md` | Testing: plan, 37 cases, results and defects |
| `docs/user_manual.md` | For operators (checking engines) and for the ML operator (running the system) |
| `docs/eda_findings.md`, `docs/data_engineering_findings.md`, `docs/model_findings.md` | What the data analysis, Spark work and model comparison found |
| `docs/monitoring_findings.md` | How the drift thresholds were measured |
| `docs/retraining.md` | The retraining loop and its measured results |
