"""
Create small CSV files that look like what an operator would upload.

They are taken from the NASA TEST sets (never used for training), with the
answer columns (rul, label) removed, so the model has not seen them.

    samples/fd001_mixed_engines.csv   10 FD001 engines: 5 truly at risk, 5 safe
    samples/fd001_short_history.csv   1 engine with only 5 cycles (shows the warning)
    samples/fd002_engines.csv         5 FD002 engines (different flight conditions:
                                      used later to demonstrate data drift)
    samples/bad_missing_column.csv    sensor s11 removed (shows a validation error)

For the data drift demo (Task 8) the drift check needs at least 3,000 rows
(params.yaml, monitoring.min_rows), so these hold whole engines up to 3,000+ rows:
    samples/drift_fd001_normal.csv    FD001 engines: same conditions as training -> no drift
    samples/drift_fd002_conditions.csv FD002 engines: six flight conditions -> drift
    samples/drift_fd003_fault.csv     FD003 engines: a second fault mode -> drift

Run from the project root:  python -m scripts.make_samples
"""

import pandas as pd

from src.config import PROJECT_ROOT, RAW_COLUMNS

INTERIM = PROJECT_ROOT / "data" / "interim"
OUT = PROJECT_ROOT / "samples"

AT_RISK_UNITS = [18, 20, 31, 34, 35]  # true remaining life at last cycle <= 30
SAFE_UNITS = [1, 2, 3, 17, 21]  # true remaining life at last cycle > 30


DRIFT_DEMO_ROWS = 3000


def first_engines_with_rows(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Whole engines, in unit order, until at least n_rows rows are collected."""
    rows_per_unit = df.groupby("unit").size().cumsum()
    last_unit = rows_per_unit[rows_per_unit >= n_rows].index[0]
    return df[df["unit"] <= last_unit]


def raw_columns_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep what a real upload would contain: no rul, label or subset."""
    return df[RAW_COLUMNS]


def main():
    OUT.mkdir(exist_ok=True)
    fd001 = pd.read_parquet(INTERIM / "test_FD001.parquet")
    fd002 = pd.read_parquet(INTERIM / "test_FD002.parquet")

    mixed = fd001[fd001["unit"].isin(AT_RISK_UNITS + SAFE_UNITS)]
    raw_columns_only(mixed).to_csv(OUT / "fd001_mixed_engines.csv", index=False)

    short = fd001[(fd001["unit"] == 31) & (fd001["cycle"] <= 5)]
    raw_columns_only(short).to_csv(OUT / "fd001_short_history.csv", index=False)

    drift = fd002[fd002["unit"].isin([1, 2, 3, 4, 5])]
    raw_columns_only(drift).to_csv(OUT / "fd002_engines.csv", index=False)

    bad = raw_columns_only(fd001[fd001["unit"] == 1]).drop(columns=["s11"])
    bad.to_csv(OUT / "bad_missing_column.csv", index=False)

    fd003 = pd.read_parquet(INTERIM / "test_FD003.parquet")
    for name, data in [
        ("drift_fd001_normal.csv", fd001),
        ("drift_fd002_conditions.csv", fd002),
        ("drift_fd003_fault.csv", fd003),
    ]:
        raw_columns_only(first_engines_with_rows(data, DRIFT_DEMO_ROWS)).to_csv(
            OUT / name, index=False
        )

    for path in sorted(OUT.glob("*.csv")):
        rows = sum(1 for _ in open(path)) - 1
        print(f"{path.name:<28} {rows:>5} rows")


if __name__ == "__main__":
    main()
