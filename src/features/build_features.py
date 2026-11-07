"""
Stage 2 of the pipeline: feature engineering with Apache Spark.

For every engine, Spark computes rolling statistics over the last `window`
cycles of each selected sensor. The work is split by engine, so Spark can
process different engines on different CPU cores at the same time.

Input : data/interim/{train,test}_FD00x.parquet   (from the ingest stage)
Output: data/features/{train,test}_FD00x.parquet  (one folder per table)
        reports/features_report.json              (row counts and speed)

Run it with:  python -m src.features.build_features
"""

import json
import os
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import PROJECT_ROOT, load_params
from src.features.feature_logic import feature_names
from src.logger import get_logger

logger = get_logger(__name__)

REPORT_PATH = PROJECT_ROOT / "reports" / "features_report.json"
ID_COLUMNS = ["subset", "unit", "cycle"]
TARGET_COLUMNS = ["rul", "label"]


def create_spark_session(spark_params: dict) -> SparkSession:
    """Start a local Spark session using the settings from params.yaml."""
    local_dir = PROJECT_ROOT / spark_params["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    # Make Spark use the same Python as this script (our virtual environment).
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName("turbofan-feature-engineering")
        .master(spark_params["master"])
        .config("spark.driver.memory", spark_params["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(spark_params["shuffle_partitions"]))
        # Spark writes temporary files here, on the project drive, instead of /tmp
        .config("spark.local.dir", str(local_dir))
        .config("spark.ui.showConsoleProgress", "false")  # no progress bars in the logs
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # hide Spark's very chatty INFO logs
    return spark


def add_rolling_features(sdf: DataFrame, sensors: list, window: int) -> DataFrame:
    """
    Add {sensor}_mean{window} and {sensor}_std{window} columns.

    The Window below says: for each row, look at rows of the SAME engine
    (partitionBy), in cycle order (orderBy), from (window - 1) rows back up to
    the current row (rowsBetween). Unit numbers repeat across subsets
    (FD001 and FD002 both have a unit 1), so we partition by subset too.
    """
    engine_window = (
        Window.partitionBy("subset", "unit")
        .orderBy("cycle")
        .rowsBetween(-(window - 1), Window.currentRow)
    )

    new_columns = []
    for s in sensors:
        new_columns.append(F.avg(s).over(engine_window).alias(f"{s}_mean{window}"))
        # stddev_samp of a single row is NULL in Spark; we use 0, same as pandas
        new_columns.append(
            F.coalesce(F.stddev_samp(s).over(engine_window), F.lit(0.0)).alias(f"{s}_std{window}")
        )
    return sdf.select("*", *new_columns)


def process_table(spark: SparkSession, in_path: Path, out_path: Path,
                  sensors: list, window: int) -> int:
    """Read one table, add features, keep only the needed columns, write it out."""
    sdf = spark.read.parquet(str(in_path))
    sdf = add_rolling_features(sdf, sensors, window)

    # Fixed column order: ids, model features, targets.
    columns = ID_COLUMNS + feature_names(sensors, window) + TARGET_COLUMNS
    sdf = sdf.select(*columns).orderBy("unit", "cycle")

    # coalesce(1) = write a single file. Our tables are small, so one file
    # is simpler for later stages than many tiny ones.
    sdf.coalesce(1).write.mode("overwrite").parquet(str(out_path))

    # Count from the written file, so the number reflects what was saved
    return spark.read.parquet(str(out_path)).count()


def main() -> None:
    params = load_params()
    feat = params["features"]
    sensors, window = feat["sensors"], int(feat["window"])
    in_dir = PROJECT_ROOT / feat["input_dir"]
    out_dir = PROJECT_ROOT / feat["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Features: %d sensors, window %d -> %d model inputs",
                len(sensors), window, len(feature_names(sensors, window)))

    start = time.perf_counter()
    spark = create_spark_session(params["spark"])
    startup_seconds = time.perf_counter() - start
    logger.info("Spark %s started in %.1f s", spark.version, startup_seconds)

    report = {"spark_version": spark.version, "sensors": len(sensors),
              "window": window, "tables": {}}
    total_rows = 0
    work_start = time.perf_counter()
    try:
        for subset in params["data"]["subsets"]:
            for split in ["train", "test"]:
                name = f"{split}_{subset}"
                t0 = time.perf_counter()
                rows = process_table(spark, in_dir / f"{name}.parquet",
                                     out_dir / f"{name}.parquet", sensors, window)
                seconds = time.perf_counter() - t0
                total_rows += rows
                report["tables"][name] = {"rows": rows, "seconds": round(seconds, 2)}
                logger.info("%s: %d rows in %.2f s", name, rows, seconds)
    except Exception as err:
        # Any Spark or file error: log it clearly and fail the DVC stage.
        logger.error("Feature engineering failed: %s", err)
        raise
    finally:
        spark.stop()  # always release Spark's memory, even after an error

    work_seconds = time.perf_counter() - work_start
    report["total_rows"] = total_rows
    report["spark_startup_seconds"] = round(startup_seconds, 2)
    report["processing_seconds"] = round(work_seconds, 2)
    report["rows_per_second"] = round(total_rows / work_seconds)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Done: %d rows in %.1f s (%d rows/s). Report: %s",
                total_rows, work_seconds, report["rows_per_second"], REPORT_PATH)


if __name__ == "__main__":
    main()
