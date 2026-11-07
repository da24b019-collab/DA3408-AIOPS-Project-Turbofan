# Data Engineering Findings (Spark feature stage)

## Feature stage (dvc stage: features)
- Tool: Apache Spark 4.2.1 (local mode, 12 cores), Java 17.
- Input: data/interim (8 tables, 265,256 rows). Output: data/features (42 features).
- Features per sensor (14 sensors): raw value, rolling mean and rolling sample std
  over the last 10 cycles, computed per engine with a Spark Window
  (partitionBy subset and unit, orderBy cycle).
- Run on 12 cores: Spark startup 3.7 s, processing 17.6 s, 15,075 rows/s
  (reports/features_report.json).

## Training-serving consistency
- The API will use a pandas implementation of the same features
  (src/features/feature_logic.py).
- tests/test_features.py::test_spark_matches_pandas checks both give the same
  values; on real data the largest difference was about 1e-6 (floating-point rounding).

## Scalability benchmark (experiments/spark_benchmark.py)
Synthetic fleets made by copying the 709 training engines with 1% noise (load test only).

| Scale | Rows | pandas (s) | Spark (s) | pandas rows/s | Spark rows/s |
|---|---|---|---|---|---|
| x1 | 160,359 | 0.51 | 1.50 | 316,445 | 106,778 |
| x5 | 801,795 | 2.64 | 3.03 | 303,531 | 264,646 |
| x20 | 3,207,180 | 19.19 | 10.75 | 167,135 | 298,475 |

- Small data: pandas is faster, because Spark's fixed coordination costs dominate.
- Crossover at about 800,000 rows; at 3.2 million rows Spark is 1.8 times faster.
- Spark throughput rises with data size; pandas throughput falls at 3.2 million rows
  (memory pressure, single core).
- On a 2-core machine Spark did not overtake pandas at these sizes, so the gain comes
  from parallelism across cores.
