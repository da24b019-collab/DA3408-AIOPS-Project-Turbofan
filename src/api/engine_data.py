"""
Reading and validating the sensor readings a user sends to the API.

The operator is not a data scientist, so bad input must produce a clear
message ("column s11 is missing") instead of a crash or, worse, a silent
wrong prediction. Every rule here is listed in docs/lld_api.md, section 5.
"""

import io

import pandas as pd

ID_COLUMNS = ["unit", "cycle"]


class InputValidationError(Exception):
    """The readings cannot be used for a prediction. The message is shown to the user."""


class BadFileError(Exception):
    """The uploaded file itself is unusable (not a CSV, empty, too large)."""


def read_csv_upload(filename: str, content: bytes, max_mb: float) -> pd.DataFrame:
    """Turn an uploaded file into a DataFrame, or raise BadFileError."""
    if not filename or not filename.lower().endswith(".csv"):
        raise BadFileError("Please upload a .csv file.")
    if len(content) == 0:
        raise BadFileError("The uploaded file is empty.")
    if len(content) > max_mb * 1024 * 1024:
        raise BadFileError(f"The file is larger than {max_mb:g} MB.")
    try:
        return pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as err:
        raise BadFileError(f"The file could not be read as CSV: {err}") from err


def readings_to_frame(readings: list) -> pd.DataFrame:
    """JSON readings (a list of dicts) to a DataFrame."""
    return pd.DataFrame(readings)


def validate_readings(df: pd.DataFrame, sensors: list, max_rows: int) -> pd.DataFrame:
    """
    Check the readings and return a clean copy with only the needed columns.
    Raises InputValidationError with a message a non-technical user can act on.
    """
    if df is None or len(df) == 0:
        raise InputValidationError("No readings were provided.")
    if len(df) > max_rows:
        raise InputValidationError(f"Too many readings: {len(df)} (the limit is {max_rows}).")

    # Column names: ignore case and surrounding spaces ("S11 " counts as "s11")
    df = df.rename(columns=lambda c: str(c).strip().lower())
    required = ID_COLUMNS + list(sensors)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputValidationError(f"Missing required column(s): {', '.join(missing)}.")
    df = df[required].copy()

    # Every value must be a number (text such as "n/a" becomes NaN here)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    bad_columns = [c for c in required if numeric[c].isna().any()]
    if bad_columns:
        raise InputValidationError(
            f"Missing or non-numeric values in column(s): {', '.join(bad_columns)}."
        )

    # unit and cycle must be positive whole numbers
    for col in ID_COLUMNS:
        values = numeric[col]
        if (values <= 0).any() or (values % 1 != 0).any():
            raise InputValidationError(f"Column '{col}' must contain positive whole numbers.")
    numeric[ID_COLUMNS] = numeric[ID_COLUMNS].astype(int)

    # Each engine can report each cycle only once
    duplicated = numeric.duplicated(subset=ID_COLUMNS)
    if duplicated.any():
        first = numeric[duplicated].iloc[0]
        raise InputValidationError(
            f"Engine {int(first['unit'])} has cycle {int(first['cycle'])} more than once."
        )

    numeric[sensors] = numeric[sensors].astype("float64")
    return numeric.sort_values(ID_COLUMNS).reset_index(drop=True)
