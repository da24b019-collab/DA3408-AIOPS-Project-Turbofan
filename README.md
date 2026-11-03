# Turbofan Predictive Maintenance (AIOps Project)

Predicts whether a turbofan engine will fail within the next 30 cycles,
using the NASA C-MAPSS dataset, built and managed with AIOps practices.

## Current status

| Stage | Status |
|---|---|
| 0. Project setup (Git, DVC, folder structure) | Done |
| 1. Data ingestion and validation | Done |
| 2. Exploratory data analysis | Next |
| 3. Feature engineering with Spark | Planned |
| 4. Model training with MLflow tracking | Planned |
| 5. Backend API (FastAPI + MLflow serving) | Planned |
| 6. Frontend dashboard | Planned |
| 7. Docker Compose | Planned |
| 8. Monitoring (Prometheus, Grafana, drift) | Planned |
| 9. Retraining, approval and pipeline console | Planned |
| 10. Tests and documentation | Ongoing |

## Folder structure

```
turbofan-pdm/
├── data/
│   ├── raw/          original NASA files (tracked by DVC, never edited)
│   └── interim/      cleaned + labelled Parquet files (output of ingest stage)
├── src/              pipeline code (one module per stage)
│   ├── config.py     column names, sensor descriptions, params loader
│   ├── logger.py     shared logging setup
│   └── data/ingest.py
├── tests/            unit tests (pytest)
├── notebooks/        EDA notebooks
├── reports/          small JSON metrics written by stages
├── backend/          FastAPI service (later)
├── frontend/         web dashboard (later)
├── monitoring/       Prometheus and Grafana config (later)
├── docs/             architecture, HLD, LLD, test plan, user manual
├── params.yaml       every tunable value
└── dvc.yaml          pipeline definition (the DVC DAG)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the 12 files `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` in `data/raw/`.

## Run the pipeline

```bash
dvc repro      # runs only the stages whose inputs changed
dvc dag        # shows the pipeline graph
pytest -v      # runs the unit tests
```
