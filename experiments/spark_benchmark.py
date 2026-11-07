"""
Scalability benchmark: pandas vs Spark for the rolling-feature computation.

Our real dataset is small, so Spark's fixed start-up and coordination costs
dominate and pandas is faster. To show how each tool behaves as data grows,
we make a synthetic "bigger fleet" by copying the 709 training engines of all
four subsets many times, giving each copy new unit numbers and a little
random noise. This is a LOAD TEST only; the copies are never used to train
the model.

Both tools read the same Parquet file and compute the same 42 features.
Spark's result goes to the "noop" output (compute everything, write nothing),
so neither tool's time includes writing results to disk.

Run from the project root:
    python -m experiments.spark_benchmark
"""

import json
import time

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import add_rolling_features, create_spark_session
from src.features.feature_logic import compute_features_pandas

SCALE_FACTORS = [1, 5, 20]  # how many copies of the fleet to test
BENCH_DIR = PROJECT_ROOT / "tmp" / "benchmark"
REPORT_PATH = PROJECT_ROOT / "reports" / "spark_benchmark.json"


def build_scaled_fleet(base: pd.DataFrame, factor: int, path) -> int:
    """Write `factor` noisy copies of the fleet to one Parquet file."""
    rng = np.random.default_rng(seed=42)
    max_unit = int(base["unit"].max())
    copies = []
    for k in range(factor):
        copy = base.copy()
        copy["unit"] = copy["unit"] + k * max_unit  # new, non-overlapping unit numbers
        noise = rng.normal(0, 0.01, size=(len(copy), len(SENSORS)))
        copy[SENSORS] = copy[SENSORS].to_numpy() * (1 + noise)  # +/- about 1%
        copies.append(copy)
    fleet = pd.concat(copies, ignore_index=True)
    fleet.to_parquet(path, index=False)
    return len(fleet)


def time_pandas(path) -> float:
    start = time.perf_counter()
    df = pd.read_parquet(path)
    compute_features_pandas(df, SENSORS, WINDOW)
    return time.perf_counter() - start


def time_spark(spark, path) -> float:
    start = time.perf_counter()
    sdf = spark.read.parquet(str(path))
    add_rolling_features(sdf, SENSORS, WINDOW).write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


params = load_params()
SENSORS = params["features"]["sensors"]
WINDOW = int(params["features"]["window"])


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    interim = PROJECT_ROOT / params["features"]["input_dir"]

    # One fleet = all four training subsets; units renumbered so they never clash
    parts, offset = [], 0
    for subset in params["data"]["subsets"]:
        df = pd.read_parquet(interim / f"train_{subset}.parquet")[["unit", "cycle"] + SENSORS]
        df["unit"] = df["unit"] + offset
        offset = int(df["unit"].max())
        parts.append(df)
    base = pd.concat(parts, ignore_index=True)
    base["subset"] = "FLEET"  # the Spark function partitions by subset and unit
    print(f"Base fleet: {base['unit'].nunique()} engines, {len(base)} rows")

    spark = create_spark_session(params["spark"])
    results = []
    try:
        for factor in SCALE_FACTORS:
            path = BENCH_DIR / f"fleet_x{factor}.parquet"
            rows = build_scaled_fleet(base, factor, path)
            time_spark(spark, path)  # warm-up run, so Spark's first-job costs are not counted
            pandas_s = time_pandas(path)
            spark_s = time_spark(spark, path)
            results.append({
                "scale": factor, "rows": rows,
                "pandas_seconds": round(pandas_s, 2), "spark_seconds": round(spark_s, 2),
                "pandas_rows_per_s": round(rows / pandas_s), "spark_rows_per_s": round(rows / spark_s),
            })
            print(f"x{factor:<3} {rows:>10,} rows | pandas {pandas_s:6.2f} s | spark {spark_s:6.2f} s")
    finally:
        spark.stop()

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
