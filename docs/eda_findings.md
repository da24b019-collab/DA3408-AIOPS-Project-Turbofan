# EDA Findings (notebooks/01_eda.ipynb)

Dataset: NASA C-MAPSS, FD001 training set (100 engines, 20,631 rows) unless stated.

## 1. Engine lifetimes vary widely (Figure 01)
- Lifetimes range from 128 to 362 cycles (mean 206.3, median 199).
- A fixed maintenance schedule safe for the shortest-lived engine would service
  long-lived engines 2 to 3 times more often than needed. This motivates
  predictive maintenance.
- A 30-cycle warning horizon is about 15% of a median lifetime.

## 2. Sensors removed
| Sensor(s) | Reason | Evidence |
|---|---|---|
| s1, s5, s10, s16, s18, s19 | Zero variance (constant in every row) | Cell 3 |
| s6 | Mostly noise: noise ratio 1.36, correlation with RUL only -0.13 | Cells 4 and 6 |
| setting_1, setting_2, setting_3 | FD001 has a single operating condition; not used as model features | Cells 3 and 7 |

Sensors kept (14): s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.

## 3. Strongest wear indicators (Figure 02)
- s11 (HPC outlet static pressure, -0.70), s4 (LPT outlet temperature, -0.68),
  s12 (fuel flow to Ps30 ratio, +0.67), s7 (HPC outlet pressure, +0.66).
- These sit closest to the HPC, which is FD001's fault mode, so the data agrees with the physics.
- s9 and s14 have weak linear correlation (-0.39, -0.31) but are the smoothest signals
  (noise ratio 0.27, 0.23). Kept, because tree models can use non-linear patterns.

## 4. Degradation pattern (Figure 03)
- Sensors drift slowly for most of an engine's life, then change sharply in the last
  50 to 75 cycles. The 30-cycle threshold falls inside this steep region.
- All engines follow the same shape, so one model can serve the whole fleet.

## 5. Rolling window size = 10 cycles (Figure 04)
- Single readings are noisy; rolling statistics expose the trend.
- Window 5 is still noisy; window 20 lags behind the final rise (late warnings).
- Window 10 balances smoothing and responsiveness, and is a third of the 30-cycle horizon.

## 6. Drift between subsets (Figures 05 and 06)
- FD001 and FD003: 1 operating condition. FD002 and FD004: 6 operating conditions.
- FD002 sensor ranges extend far below FD001 (s7: FD001 549.9 to 556.1, FD002 136.8 to 555.8).
- Replaying FD002/FD004 into the system demonstrates covariate drift, which PSI should detect.
- FD003 has a second fault mode (fan) with the same operating condition: closer to concept
  drift, which input-based PSI may not detect. Model performance monitoring is needed for that.
## Correction (Task 8)

The EDA compared FD003 with FD001 by sensor averages and concluded FD003 would not be
detectable as drift. Comparing the full distribution of each sensor (PSI, 10 bins) shows
otherwise: windows of 3,000 to 5,000 FD003 readings have a highest sensor PSI above the
alarm threshold of 1.0 in 87 to 91% of cases, while normal FD001 data stays at or below
it in 99 to 100% of cases. Details: docs/monitoring_findings.md.
