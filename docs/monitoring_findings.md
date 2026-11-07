# Monitoring findings (Task 8)

## 1. What is monitored, and by which tool

| What | Tool | How it gets the data |
|---|---|---|
| Backend: requests, errors, latency, predictions, model state, data drift | Prometheus client in FastAPI (`/metrics`) | Prometheus pulls every 15 s |
| MLflow tracking server: requests and their timing | `mlflow server --expose-prometheus` | Prometheus pulls |
| Pipeline stages (features, train): run time, success, results | Pushgateway | Each DVC stage pushes when it ends; Prometheus pulls from the Pushgateway |
| Every container: CPU, memory, network | cAdvisor | Prometheus pulls |
| The laptop: CPU, memory, disk | node exporter | Prometheus pulls |
| Is each service answering? | Blackbox exporter | Prometheus asks it to call each health URL |
| Alerts | Prometheus rules → Alertmanager → backend `/alerts/webhook` | Push |
| Dashboards | Grafana, provisioned from files in `monitoring/grafana` | Queries Prometheus |

**Why a Pushgateway for the pipeline:** Prometheus pulls, but a DVC stage runs for under a
minute and exits, so there is nothing to pull from. The stage pushes its numbers to the
Pushgateway before exiting, and the Pushgateway keeps them for Prometheus.

## 2. Data drift: method

**Population Stability Index (PSI)**, per sensor, on the 14 raw sensors the model uses:

1. At training time, each sensor's training readings (FD001) are cut into 10 bins at their
   deciles, so each bin holds 10% of them. The bin edges and fractions are saved as
   `drift/baseline.json` in the model's MLflow run.
2. The backend keeps the latest 5,000 readings it received for prediction.
3. For each sensor: PSI = Σ (actual − expected) · ln(actual / expected) over the bins
   (empty bins are given a fraction of 0.0001 so the logarithm stays finite).
4. Drift is reported when the **highest** of the 14 PSI values is above the threshold.

PSI was chosen over a statistical test (for example Kolmogorov-Smirnov) because with
thousands of rows such a test calls even tiny, harmless differences "significant", and
because PSI is a single number per sensor that is easy to plot and explain.

## 3. Choosing the threshold and window size

Experiment: `python -m experiments.drift_threshold_study`. For each case, 100 random windows
of **whole engines** (as uploads arrive: one engine's history at a time) are scored.

- Normal data: FD001 test engines (same conditions and fault as training).
- Drift: FD003 (adds a second fault mode) and FD002 (six operating conditions instead of one).

```
case                            rows    min median    p95    max  above threshold
FD001 test (normal)               50   2.65   4.95   7.13   8.29            100%
FD003 test (new fault)            50   3.19   6.41   8.58   9.40            100%
FD002 test (new conditions)       50   5.03   7.05   8.39   8.85            100%

FD001 test (normal)              500   0.28   1.27   2.97   4.03             82%
FD003 test (new fault)           500   0.55   2.30   4.47   6.38             94%
FD002 test (new conditions)      500   4.20   4.74   5.57   6.10            100%

FD001 test (normal)             1000   0.18   0.99   1.61   2.40             49%
FD003 test (new fault)          1000   0.49   1.67   3.14   4.90             84%
FD002 test (new conditions)     1000   4.25   4.69   4.94   5.08            100%

FD001 test (normal)             3000   0.21   0.45   0.82   1.04              1%
FD003 test (new fault)          3000   0.76   1.44   2.15   2.85             87%
FD002 test (new conditions)     3000   4.47   4.66   4.79   4.87            100%

FD001 test (normal)             5000   0.21   0.37   0.58   0.94              0%
FD003 test (new fault)          5000   0.63   1.35   1.95   2.38             91%
FD002 test (new conditions)     5000   4.46   4.64   4.75   4.84            100%
```

**Findings**

1. **The textbook threshold of 0.25 does not fit this data.** Even normal FD001 test data
   never scores below 0.18, so a 0.25 rule would raise a drift alarm almost all the time.
   One reason: test engines are cut off earlier in their life than training engines (which
   run to failure), so their readings sit more in the "healthy" bins.
2. **Small samples inflate PSI.** With 50 rows, normal data scores 2.6 to 8.3, higher than
   real drift at larger sizes. A few engines are also highly correlated: the cycles of one
   engine are not independent samples. So drift is only judged from **3,000 rows**.
3. **How the windows are sampled matters.** An earlier quick check sampled random single
   rows from all engines, which made 500 to 1,000 rows look safe (normal max 0.6). Sampling
   whole engines, which is how data really arrives, gives 49% false alarms at 1,000 rows.
   The settings were changed to the whole-engine results.
4. **FD003 drift is detectable, but not always.** The EDA said FD003 looks like FD001 on
   average. Per-sensor distributions still differ enough to cross 1.0 in 87 to 91% of
   windows. It is missed sometimes because FD003 mixes two fault modes, one of them the same
   HPC fault as FD001, so a window made mostly of such engines really is like the training data.
5. **FD002 is always detected** (PSI about 4.6, with sensor s17 the most changed): six
   operating conditions move every sensor.

**Chosen settings** (params.yaml, `monitoring`): threshold **1.0**, judged from **3,000**
rows, sliding window of the latest **5,000** rows. Expected behaviour: 0 to 1% false alarms
on normal data, FD002 always detected, FD003 detected about 9 times in 10.

## 4. Demonstration files

`python -m scripts.make_samples` writes three upload files of whole engines (just over
3,000 rows each). Uploaded one at a time to the running backend, with the drift window
emptied in between:

| File | Rows | State | Highest PSI | Sensor |
|---|---|---|---|---|
| `samples/drift_fd001_normal.csv` | 3,062 | ok | 0.85 | s11 |
| `samples/drift_fd003_fault.csv` | 3,097 | drift | 1.15 | s11 |
| `samples/drift_fd002_conditions.csv` | 3,054 | drift | 4.58 | s17 |

Note the normal file's 0.85 is close to the threshold: these are only about 25 engines, the
smallest amount the alarm accepts. The FD003 file is only just above the threshold.

## 5. Alert rules

`monitoring/prometheus/alert_rules.yml`:

| Alert | Condition | For | Severity |
|---|---|---|---|
| ServiceDown | an uptime check fails | 1 min | critical |
| ModelNotLoaded | `pdm_model_loaded == 0` | 1 min | critical |
| HighErrorRate | more than 5% of requests answered with 5xx | 2 min | critical |
| ModelServerOutOfSync | model server serves another version | 2 min | warning |
| HighPredictionLatency | 95th percentile of `/predict*` above 1 s | 5 min | warning |
| DataDrift | `pdm_drift_detected == 1` | 1 min | warning |
| PipelineStageFailed | a stage's last push reported failure | at once | warning |
| ContainerMemoryHigh | a container above 90% of its memory limit | 5 min | warning |
| DiskAlmostFull | under 10% free on a disk | 10 min | warning |

Alertmanager groups alerts by name, sends them to the backend, and holds back
HighErrorRate and HighPredictionLatency while the model server's uptime check fails
(those follow from the model server being down, which is the alert that matters).

**Tested end to end** before deployment, with the real Prometheus, Alertmanager and
blackbox exporter binaries against the running backend: the DataDrift alert fired after
uploading the FD002 file, reached `/alerts/webhook` with the Bearer key, and a "resolved"
notification arrived after `POST /drift/reset`.
