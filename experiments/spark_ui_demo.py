"""
Keep Spark's monitoring page (the Spark UI) open for screenshots.

The Spark UI only exists while a Spark session is running. Our pipeline
stops Spark as soon as it finishes, so the page disappears within seconds.
This script runs the same feature computation on every training table and
then waits for you to press Enter before stopping Spark.

Run from the project root:
    python -m experiments.spark_ui_demo
"""

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import add_rolling_features, create_spark_session


def main():
    params = load_params()
    sensors = params["features"]["sensors"]
    window = int(params["features"]["window"])
    interim = PROJECT_ROOT / params["features"]["input_dir"]

    spark = create_spark_session(params["spark"])
    try:
        print("\nSpark UI is running at:", spark.sparkContext.uiWebUrl)
        for subset in params["data"]["subsets"]:
            sdf = spark.read.parquet(str(interim / f"train_{subset}.parquet"))
            # "noop" = compute everything but write nothing (we only want the job to show up)
            add_rolling_features(sdf, sensors, window).write.format("noop").mode("overwrite").save()
            print(f"Finished job for train_{subset}")
        input(
            "\nAll jobs done. Open the Spark UI in Firefox now.\n"
            "Press Enter here when you have finished taking screenshots... "
        )
    finally:
        spark.stop()
        print("Spark stopped.")


if __name__ == "__main__":
    main()
