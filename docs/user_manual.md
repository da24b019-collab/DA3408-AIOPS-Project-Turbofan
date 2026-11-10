# User Manual: Engine Health Dashboard

The manual has two parts:

- **Part A, for maintenance operators**: checking engines and reading the results. No
  technical knowledge is needed.
- **Part B, for the ML operator (engineer)**: starting the system, watching it, approving
  new models and rolling back.

The dashboard also has a short version of Part A built in: the **Help** page.

---

# Part A: Maintenance operator

## A1. What the dashboard tells you

For each engine you upload, the dashboard estimates the **chance that the engine fails
within the next 30 operating cycles**, and shows which sensors made it think so. Use it to
decide which engines to inspect first. It supports your decision; it does not replace the
maintenance rules of your organisation.

Open the dashboard at **http://localhost:8501**. The menu on the left has five pages:

| Page | Use it to |
|---|---|
| **Engine check** | Upload readings and see each engine's risk (your main page) |
| **Pipeline** | See how the model was built (for engineers) |
| **Monitoring** | Check that today's data still looks like the data the model learned from |
| **Model** | See which model is in use and how good it is |
| **Help** | Read these instructions and download a file template |

The **System status** box in the left menu says "Ready · model version N" when everything
works. If it says something else, see section A6.

## A2. Preparing the file

A CSV file (comma-separated, for example saved from Excel as "CSV"), **one row per engine
per cycle**:

| Column | Meaning |
|---|---|
| `unit` | Engine number |
| `cycle` | Operating cycle number (1, 2, 3, ...); each cycle only once per engine |
| `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21` | The 14 sensor readings |

- Other columns are ignored; upper or lower case in column names does not matter.
- Give at least the **last 10 cycles** of each engine. Fewer still works, but the result is
  less reliable and you will see a warning.
- Limits: 5 MB per file and 50,000 rows.
- The **Help** page has a **template** with the right columns to download.

## A3. Checking engines

1. Open **Engine check**.
2. Click **Browse files** and choose your CSV, then click **Check engines**.
   (To try the dashboard without a file, click **Try with example data**.)
3. Read the results, from top to bottom:

**Fleet summary**: how many engines are high, medium and low risk.

**All engines, riskiest first**: one row per engine:

| Risk | Meaning | Suggested action |
|---|---|---|
| **High** | Likely to fail within 30 cycles | Plan an inspection soon |
| **Medium** | Showing some signs of wear | Keep a close eye; check again with new readings |
| **Low** | Readings look like a healthy engine | No action needed |

The **failure probability** is the model's estimate (for example "42%"). Very high or very
low values are shown as "over 99%" or "under 1%": the model is never certain.

Click **Download these results (CSV)** to save the table.

**Why this risk?**: choose an engine to see a one-sentence summary and the sensors that
pushed its risk up or down the most (for example "HPC outlet temperature rising").

## A4. Messages you may see

| Message | What to do |
|---|---|
| "Only N cycle(s) received; at least 10 give a more reliable prediction" | The result is shown but less reliable. Add more history if you have it. |
| "Missing required column(s): ..." | Add the listed sensor columns (use the Help page template). |
| "Missing or non-numeric values in column(s): ..." | Fill the empty cells; remove text such as "n/a" from number columns. |
| "Engine X has cycle Y more than once." | Remove the duplicate row. |
| "Please upload a .csv file." | Save the file as CSV (comma-separated values) first. |
| "The file is larger than 5 MB." / "Too many readings" | Split the file into smaller files. |
| "The dashboard's API key does not match the backend's." | Contact the ML operator (configuration problem). |
| "The prediction service is not running or not reachable." | Contact the ML operator. |

## A5. The Monitoring page (should I trust today's results?)

The model learned from engines flying in particular conditions. If your engines fly in
very different conditions, its answers can be wrong without any error message.

| Box colour | Meaning |
|---|---|
| Blue, "Collecting readings" | Fewer than 3,000 readings received so far; the check has not started |
| Green, "No drift" | Your data looks like the training data: results can be trusted |
| Red, "Data drift detected" | Your data looks different: treat results with caution and tell the ML operator |

When drift is detected, the system starts training an improved model by itself; an
engineer must approve it before it is used.

## A6. When something does not work

| Problem | What to do |
|---|---|
| Page does not open | Ask the ML operator whether the system is running |
| System status is not "Ready" | Wait one minute (it may be starting) and refresh; if it persists, tell the ML operator |
| Result looks wrong for an engine you know well | Check the Monitoring page; report it with the file you used |

---

# Part B: ML operator

## B1. Addresses

| Service | Address | Login |
|---|---|---|
| Dashboard | http://localhost:8501 | none (uses the API key internally) |
| Backend API, interactive docs | http://localhost:8000/docs | API key (`X-API-Key`) |
| MLflow | http://localhost:5000 | none |
| Grafana | http://localhost:3000 | `admin` / password in `.env` |
| Prometheus | http://localhost:9090 | none |
| Alertmanager | http://localhost:9093 | none |
| cAdvisor | http://localhost:8080 | none |

All services listen on 127.0.0.1 only, so they are reachable from this laptop alone.

## B2. Starting and stopping

```bash
cd /aiops-storage/AIOps/turbofan-pdm
docker compose up -d             # start all 11 containers
docker compose ps                # mlflow, model-server, backend, frontend must be "healthy"
docker compose down              # stop (data and history are kept)
```

In a **second terminal**, start the retraining worker and leave it running:

```bash
cd /aiops-storage/AIOps/turbofan-pdm && source .venv/bin/activate
python -m src.retraining.worker
```

First-time setup on a new machine is described in `README.md`.

## B3. Running the pipeline by hand

```bash
source .venv/bin/activate
dvc repro            # reruns only the stages whose inputs changed
dvc push             # store data and model versions in the DVC remote
git add dvc.lock reports && git commit -m "..."
```

The new model is registered as **challenger**. It is not used until approved (B5).
Commit code changes **before** training, so the run records a clean Git commit.

## B4. Watching the system

- **Grafana**, dashboard "Turbofan Predictive Maintenance": service health, API traffic and
  errors, predictions, data drift, pipeline results, containers and the laptop, retraining jobs.
- **Prometheus → Alerts**: the 11 rules and their state.
- **Dashboard → Monitoring**: drift state, recent alerts (with what the backend did about each),
  and the retraining jobs.

| Alert | Meaning | First thing to do |
|---|---|---|
| ServiceDown | A service does not answer | `docker compose ps`; `docker compose logs --tail 50 <name>` |
| ModelNotLoaded | Backend has no model | Check MLflow is up and a champion exists; `POST /model/reload` |
| ModelServerOutOfSync | Model server serves another version | Check the worker is running; else `docker compose restart model-server` |
| HighErrorRate / HighPredictionLatency | Requests fail or are slow | Backend and model-server logs; Grafana containers row |
| DataDrift | Incoming data differs from training data | A retrain job starts automatically; review and approve (B5) |
| PipelineStageFailed / RetrainingJobFailed | A stage or job failed | `logs/pipeline.log`, `logs/retraining.log`, the job's Result column |
| RetrainingJobStuck | A job has waited 30 minutes | Start the worker (B2) |
| ContainerMemoryHigh / DiskAlmostFull | Resources running out | Raise `mem_limit` in `docker-compose.yml` / free disk space |

## B5. Approving a new model

1. Dashboard → **Model** → **New model waiting for approval**.
2. Read the table: recall and PR-AUC of the current model (champion) and the new one
   (challenger) on the same NASA test engines, for FD001 to FD004.
3. Read the checks. If a check fails (the new model is clearly worse on FD001), approval
   is blocked.
4. Tick **I have reviewed version N ...** and click **Approve version N**.
5. The worker runs a deploy job (about 30 s): the model server restarts, the backend reloads,
   and both are checked to serve the same version. Follow it on **Monitoring → Retraining jobs**.

From the terminal instead: `python -m src.models.promote --approve`, then
`docker compose restart model-server` (the command line does not create a deploy job).

## B6. Rolling back

Dashboard → **Model** → **Roll back**: choose the version and click **Roll back to this version**.
A deploy job switches the system back. No checks are applied, because speed matters.
The newer version stays in the registry and can be approved again later.

Command line: `python -m src.models.promote --rollback N`, then restart the model server.

## B7. Starting a retraining job by hand

Dashboard → **Monitoring** → **Start a retraining job by hand**: choose the datasets and
click **Queue retraining job**. The worker refuses to start if the Git project has
uncommitted changes; commit them first.

## B8. Logs and data

| Where | What |
|---|---|
| `logs/api.log`, `logs/pipeline.log`, `logs/retraining.log` | Application logs (also shown on the Pipeline page) |
| `docker compose logs <service>` | Container output |
| `retraining/jobs.json` | All retrain and deploy jobs |
| MLflow | Every training run, its metrics, plots, tags and drift baseline |
| `/aiops-storage/dvc-storage` | DVC remote: every version of the data and models |
