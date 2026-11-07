# Retraining loop with human approval (Task 9)

## 1. The loop

```
 FD002 data arrives ──> backend drift check (PSI) ──> Prometheus ──> DataDrift alert
                                                                         │
                                                       Alertmanager webhook
                                                                         ▼
                                              backend queues a RETRAIN job
                                                                         │
             retraining worker (laptop) claims it ◄──────────────────────┘
               1. git must be clean
               2. params.yaml train.subsets = [FD001, FD002], commit
               3. dvc repro -f -s train  ──> MLflow: new version, alias "challenger"
               4. commit dvc.lock and reports
                                                                         │
             Model page: champion vs challenger on the same test engines ◄┘
               approval gate (FD001 must not get worse by more than 0.05)
               person ticks "I have reviewed" and clicks Approve
                                                                         │
             backend moves alias "champion", queues a DEPLOY job ◄──────┘
               worker: docker compose restart model-server, wait for /ping,
               POST /model/reload, check backend and model server serve the same version
```

Rollback: Model page, pick an earlier version, Roll back. The alias moves at once and a
deploy job restarts the model server. No checks are applied, because speed matters in an
emergency, and the version was already approved when it was first used.

## 2. Design decisions

| Decision | Reason |
|---|---|
| The backend only queues jobs; a worker on the laptop runs them | Training needs Spark, DVC and Git; deploying needs Docker. Giving the backend container the Docker socket would let anyone who breaks into the API control every container. The worker pulls work, like a CI runner. |
| Jobs stored in `retraining/jobs.json` (not in Git) | Survives a backend restart. Written to a temporary file and renamed, so a crash cannot leave half a file. |
| One open retrain job at a time, plus a 30-minute cooldown for alert-triggered jobs | A DataDrift alert repeats every hour while it fires. Without these, each repeat would start another training run. A person can still start a job by hand. |
| The worker refuses to run with uncommitted changes, and commits the params change and the results | Every model version can be traced to an exact commit (`git_commit` tag, `git_dirty=False`). |
| A failed training run is undone (`git reset --hard` to the starting commit, `dvc checkout`) | DVC deletes a stage's outputs before it runs, so a failure would leave the project half-changed and block the next job. The tree was clean at the start, so nothing of the user's is lost. |
| New model is a challenger until a person approves | The rubric asks for human approval. A model can look better on paper and still be worse for the original use. |
| The approval gate compares both models live, on the same test engines | The test score stored with each run comes from different test sets (FD001 vs FD001+FD002), so they cannot be compared directly. |
| One drift baseline per training dataset | A single baseline of FD001+FD002 mixed would call pure FD001 data "drifted" (PSI about 4.9), because FD001 fills only a part of the mixed bins. With one baseline each, new data is judged against the dataset it is closest to. |

## 3. Measured results

Champion trained on FD001 vs challenger trained on FD001+FD002, scored on the last cycle
of every NASA test engine (each with its own threshold):

| Test engines | Recall champion → challenger | PR-AUC champion → challenger |
|---|---|---|
| FD001 (original task) | 0.800 → 0.760 | 0.938 → 0.943 |
| FD002 (6 flight conditions) | **0.000 → 0.967** | 0.432 → 0.984 |
| FD003 (2 faults) | 0.300 → 0.250 | 0.735 → 0.741 |
| FD004 (6 conditions, 2 faults) | 0.000 → 0.679 | 0.296 → 0.882 |

- The FD001-only model catches **none** of the failing FD002 engines, which is what the
  DataDrift alert warned about. The retrained model catches 97%.
- On FD001, recall drops by 0.04 (one engine fewer out of 25 at risk), within the 0.05
  allowed, and PR-AUC improves. The gate passes. The cause is the higher threshold
  (0.68 instead of 0.53) needed to keep 90% recall over both datasets.
- FD003 stays poor for both: neither model has seen the second fault. That would need
  FD003 training data, which is a separate retraining decision.

Drift after deploying the retrained model (one upload each, window emptied in between):

| Upload | Before (FD001 baseline only) | After (FD001 and FD002 baselines) |
|---|---|---|
| `drift_fd001_normal.csv` | ok, 0.85 | ok, 0.85 (closest: FD001) |
| `drift_fd002_conditions.csv` | **drift, 4.58** | ok, 0.13 (closest: FD002) |
| `drift_fd003_fault.csv` | drift, 1.15 | drift, 1.15 (closest: FD001) |

The loop closes: after retraining, FD002 data is no longer reported as drift, and FD003
still is, correctly, because the model has still not been trained on it.

## 4. Known limits

- The worker must be running (`python -m src.retraining.worker`). If it is not, jobs wait;
  the RetrainingJobStuck alert fires after 30 minutes.
- Which data to retrain on is set in params.yaml (`retraining.subsets_on_drift`). In a real
  fleet, newly collected and labelled data would be used; here the FD002 training set
  plays that part.
- One window mixing FD001 and FD002 rows matches neither baseline well and may be reported
  as drift.
- Job counts in Prometheus restart from zero when the backend restarts (the jobs file does not).
