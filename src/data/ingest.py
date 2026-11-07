"""
Stage 1 of the pipeline: data ingestion and validation.

What this stage does, for each C-MAPSS subset (FD001 ... FD004):
    1. Read the raw space-separated text files and give the columns names.
    2. Validate them (right shape, no missing values, sensible cycle numbers).
       If the data is broken, we stop here instead of training on garbage.
    3. Work out the Remaining Useful Life (RUL) of every row.
    4. Turn RUL into the binary target: 1 if the engine fails within
       `horizon` cycles, else 0.
    5. Save clean tables to data/interim/ as Parquet files and write a short
       JSON report with row counts and label balance.

Run it with:  python -m src.data.ingest
(DVC runs exactly this command for the "ingest" stage.)
"""

import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, RAW_COLUMNS, load_params
from src.logger import get_logger

logger = get_logger(__name__)

REPORT_PATH = PROJECT_ROOT / "reports" / "ingest_report.json"


class DataValidationError(Exception):
    """Raised when a raw file does not look like valid C-MAPSS data."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_cmapss_file(path: Path) -> pd.DataFrame:
    """
    Read one train_/test_ file into a DataFrame with named columns.

    The files are separated by one or more spaces and every line ends with
    trailing spaces. Splitting on single spaces would create two empty extra
    columns, so we use the regex separator r"\\s+" (any run of whitespace).
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] != len(RAW_COLUMNS):
        raise DataValidationError(
            f"{path.name}: expected {len(RAW_COLUMNS)} columns, found {df.shape[1]}"
        )

    df.columns = RAW_COLUMNS
    # unit and cycle are whole numbers; the rest stay as floats.
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def read_rul_file(path: Path) -> pd.Series:
    """
    Read a RUL_ file. It has one number per line: the true remaining life of
    each test engine at its LAST recorded cycle. Line 1 is engine 1, and so on.
    """
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    rul = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    # Shift the index so it starts at 1, matching the unit numbers.
    rul.index = range(1, len(rul) + 1)
    return rul.astype(int)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_frame(df: pd.DataFrame, name: str) -> None:
    """
    Basic data quality checks. Each check protects against a specific way
    the data could be broken. Raises DataValidationError on the first problem.
    """
    # 1. Missing values would break the rolling-window features later.
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        raise DataValidationError(f"{name}: contains {null_count} missing values")

    # 2. Unit numbers and cycle numbers must be positive.
    if (df["unit"] < 1).any() or (df["cycle"] < 1).any():
        raise DataValidationError(f"{name}: found unit or cycle numbers below 1")

    # 3. For each engine, cycles must be 1, 2, 3, ... with no gaps or repeats.
    #    A gap would mean a lost reading, and our RUL formula (max cycle minus
    #    current cycle) assumes the history is complete.
    for unit, group in df.groupby("unit"):
        cycles = group["cycle"].to_numpy()
        expected = list(range(1, len(cycles) + 1))
        if list(cycles) != expected:
            raise DataValidationError(
                f"{name}: unit {unit} has missing, repeated or unordered cycles"
            )

    logger.info(
        "%s passed validation: %d rows, %d engines",
        name,
        len(df),
        df["unit"].nunique(),
    )


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def add_train_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Training engines run until they fail, so the last cycle of each engine
    is its failure point. Remaining life at any row is therefore:

        RUL = (last cycle of this engine) - (current cycle)

    Example: an engine with 192 cycles has RUL 191 at cycle 1 and RUL 0 at 192.
    """
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last_cycle - df["cycle"]
    df["label"] = (df["rul"] <= horizon).astype(int)
    return df


def add_test_labels(df: pd.DataFrame, true_rul: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Test engines are cut off BEFORE they fail. NASA gives the true RUL at the
    last recorded cycle, so for earlier rows we add the cycles still to go:

        RUL = (true RUL at last cycle) + (last cycle - current cycle)
    """
    n_units = df["unit"].nunique()
    if len(true_rul) != n_units:
        raise DataValidationError(
            f"RUL file has {len(true_rul)} values but test data has {n_units} engines"
        )

    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    rul_at_end = df["unit"].map(true_rul)
    df["rul"] = rul_at_end + (last_cycle - df["cycle"])
    df["label"] = (df["rul"] <= horizon).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main stage
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> dict:
    """Small summary for the report: size and class balance."""
    return {
        "rows": int(len(df)),
        "engines": int(df["unit"].nunique()),
        "positive_rate": round(float(df["label"].mean()), 4),
    }


def ingest_subset(subset: str, raw_dir: Path, out_dir: Path, horizon: int) -> dict:
    """Ingest the train and test files of one subset (e.g. "FD001")."""
    logger.info("Ingesting subset %s", subset)

    train = read_cmapss_file(raw_dir / f"train_{subset}.txt")
    validate_frame(train, f"train_{subset}")
    train = add_train_labels(train, horizon)

    test = read_cmapss_file(raw_dir / f"test_{subset}.txt")
    validate_frame(test, f"test_{subset}")
    true_rul = read_rul_file(raw_dir / f"RUL_{subset}.txt")
    test = add_test_labels(test, true_rul, horizon)

    # Remember which subset each row came from. This matters later when we
    # mix subsets for drift replay and retraining.
    train["subset"] = subset
    test["subset"] = subset

    # Parquet keeps column types, is compressed, and Spark reads it natively.
    train.to_parquet(out_dir / f"train_{subset}.parquet", index=False)
    test.to_parquet(out_dir / f"test_{subset}.parquet", index=False)

    return {"train": summarise(train), "test": summarise(test)}


def main() -> None:
    params = load_params()
    raw_dir = PROJECT_ROOT / params["data"]["raw_dir"]
    out_dir = PROJECT_ROOT / params["data"]["interim_dir"]
    horizon = int(params["label"]["horizon"])
    subsets = params["data"]["subsets"]

    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {"horizon": horizon, "subsets": {}}
    try:
        for subset in subsets:
            report["subsets"][subset] = ingest_subset(subset, raw_dir, out_dir, horizon)
    except (FileNotFoundError, DataValidationError) as err:
        # Log the reason clearly, then re-raise so the process exits with an
        # error code. DVC sees the failure and does not mark the stage as done.
        logger.error("Ingestion failed: %s", err)
        raise

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Ingestion finished. Report written to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
