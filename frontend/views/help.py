"""
"Help" page: a short user manual inside the dashboard itself.
"""

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError

# Used only if the backend cannot be asked for the current sensor list
DEFAULT_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
                   "s13", "s14", "s15", "s17", "s20", "s21"]


def required_sensors() -> list:
    try:
        return [s["code"] for s in ApiClient().model_info()["sensors"]]
    except ApiError:
        return DEFAULT_SENSORS


def template_csv(sensors: list) -> bytes:
    """An empty file with the right columns and one example row."""
    example = {"unit": 1, "cycle": 1, **{s: 0.0 for s in sensors}}
    return pd.DataFrame([example]).to_csv(index=False).encode()


def render() -> None:
    st.title("Help")

    st.subheader("What this dashboard does")
    st.markdown(
        "It estimates, for each engine, the chance that it will **fail within the next "
        "30 operating cycles**, using its recent sensor readings. Engines are sorted so the "
        "riskiest come first, and each result explains which sensors drove it.")

    st.subheader("How to use it")
    st.markdown(
        "1. Open **Engine check**.\n"
        "2. Upload a CSV file with your engines' readings (format below), then press "
        "**Check engines**. To see how it works first, press **Try with example data**.\n"
        "3. Read the **fleet summary**: how many engines are high, medium or low risk.\n"
        "4. In the table, start with the engines at the top.\n"
        "5. Under **Why this risk?**, choose an engine to see the sensors behind its result.\n"
        "6. Press **Download these results** to keep a copy.")

    st.subheader("What the risk levels mean")
    st.markdown(
        "- 🔴 **High**: likely to fail within 30 cycles. Schedule an inspection now.\n"
        "- 🟠 **Medium**: some warning signs. Watch it closely and check it again soon.\n"
        "- 🟢 **Low**: readings look like a healthy engine.\n\n"
        "The model catches most, but not all, failing engines, so treat **Medium** seriously.")

    st.subheader("File format")
    sensors = required_sensors()
    st.markdown(
        "A **CSV file** (comma-separated, with a header row), **one row per engine per "
        "cycle**. Required columns:\n"
        "- `unit`: the engine number (a whole number, 1 or more)\n"
        "- `cycle`: the operating cycle (a whole number, 1 or more, each only once per engine)\n"
        f"- the sensor readings: `{'`, `'.join(sensors)}`\n\n"
        "Other columns are ignored. Upper or lower case column names both work. "
        "Send **at least 10 cycles per engine** for a reliable result.")
    st.download_button("Download a template file", template_csv(sensors),
                       file_name="engine_readings_template.csv", mime="text/csv")

    st.subheader("If something goes wrong")
    st.markdown(
        "| Message | What to do |\n|---|---|\n"
        "| *Missing required column(s): …* | Add the named columns to your file. |\n"
        "| *Missing or non-numeric values in …* | Every reading must be a number; "
        "remove text such as `n/a` and fill empty cells. |\n"
        "| *Engine X has cycle Y more than once* | Remove the duplicated row. |\n"
        "| *Please upload a .csv file* | Save your spreadsheet as CSV first. |\n"
        "| *Cannot reach the prediction service* | The system is not running; "
        "contact the administrator. |")
