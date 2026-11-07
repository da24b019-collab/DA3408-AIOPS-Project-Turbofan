"""
"Engine check" page: upload sensor readings, see each engine's failure risk.

Flow for the operator:
    1. upload a CSV (or try the example file)
    2. read the fleet summary: how many engines are high / medium / low risk
    3. look at the ranked table, riskiest engines first
    4. open one engine to see WHY it got its risk (top sensors)
"""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError
from ui import RISK_COLOURS, risk_badge, risk_label, show_api_error

SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", Path(__file__).resolve().parents[2] / "samples"))
EXAMPLE_FILE = SAMPLES_DIR / "fd001_mixed_engines.csv"


def run_prediction(filename: str, content: bytes) -> None:
    """Send the file to the backend and keep the answer for this browser session."""
    with st.spinner("Checking engines..."):
        try:
            st.session_state["prediction"] = ApiClient().predict_csv(filename, content)
            st.session_state["prediction_file"] = filename
            st.session_state.pop("prediction_error", None)
        except ApiError as err:
            st.session_state["prediction_error"] = err
            st.session_state.pop("prediction", None)


def results_table(result: dict) -> pd.DataFrame:
    rows = []
    for e in result["engines"]:
        top = e["explanation"]["top_factors"]
        main_reason = next((f["description"] for f in top if f["effect"] == "raises risk"), "-")
        rows.append({
            "Engine": e["unit"],
            "Risk": risk_label(e["risk_level"]),
            "Failure probability": e["failure_probability"],
            "Cycles received": e["cycles_received"],
            "Main reason": main_reason if e["risk_level"] != "low" else "-",
            "Notes": "⚠ " + e["warnings"][0] if e["warnings"] else "",
        })
    table = pd.DataFrame(rows)
    return table.sort_values("Failure probability", ascending=False).reset_index(drop=True)


def probability_text(probability: float) -> str:
    """A model is never certain, so very high or low values are shown as ranges."""
    if probability >= 0.995:
        return "over 99%"
    if probability < 0.005:
        return "under 1%"
    return f"{probability:.0%}"


def factor_chart(factors: list) -> alt.Chart:
    """Bar chart of the sensors behind one prediction (red = raises risk, green = lowers it)."""
    data = pd.DataFrame(factors)
    return (alt.Chart(data)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("contribution:Q", title="Effect on risk (SHAP value)"),
                y=alt.Y("description:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=260)),  # show full sensor names
                color=alt.Color("effect:N", title=None,
                                scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                range=["#C0392B", "#2E8B57"])),
                tooltip=["description", "effect", alt.Tooltip("contribution:Q", format=".3f")])
            .properties(height=60 + 40 * len(data)))


def show_results(result: dict) -> None:
    engines = result["engines"]
    counts = {level: sum(e["risk_level"] == level for e in engines)
              for level in ("high", "medium", "low")}

    st.subheader("Fleet summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines checked", len(engines))
    c2.metric("🔴 High risk", counts["high"], help="Likely to fail within 30 cycles: inspect now.")
    c3.metric("🟠 Medium risk", counts["medium"], help="Worth watching closely.")
    c4.metric("🟢 Low risk", counts["low"])
    st.caption(f"File: {st.session_state.get('prediction_file', '')} · model version "
               f"{result['model_version']} · an engine is high risk when its failure "
               f"probability is at least {result['threshold']:.2f}")

    st.subheader("All engines, riskiest first")
    table = results_table(result)
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "Failure probability": st.column_config.ProgressColumn(
                "Failure probability", min_value=0.0, max_value=1.0, format="%.2f",
                help="Model's estimated chance of failure within the next 30 cycles"),
            "Engine": st.column_config.NumberColumn("Engine", format="%d"),
        })
    st.download_button("Download these results (CSV)", table.to_csv(index=False).encode(),
                       file_name="engine_risk_results.csv", mime="text/csv")

    st.subheader("Why this risk?")
    by_unit = {e["unit"]: e for e in engines}
    chosen = st.selectbox("Choose an engine", list(table["Engine"]),
                          format_func=lambda u: f"Engine {u}")
    engine = by_unit[chosen]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"#### Engine {engine['unit']} &nbsp; {risk_badge(engine['risk_level'])}",
                    unsafe_allow_html=True)
        st.metric("Failure probability", probability_text(engine["failure_probability"]))
        st.markdown(f"<p class='muted'>Based on cycles up to {engine['last_cycle']} "
                    f"({engine['cycles_received']} received)</p>", unsafe_allow_html=True)
    with right:
        colours = RISK_COLOURS[engine["risk_level"]]
        st.markdown(f"<div style='background:{colours['background']};color:{colours['text']};"
                    f"padding:12px 16px;border-radius:10px'>{engine['explanation']['summary']}"
                    "</div>", unsafe_allow_html=True)
        for warning in engine["warnings"]:
            st.warning(warning)
        if engine["explanation"]["top_factors"]:
            st.altair_chart(factor_chart(engine["explanation"]["top_factors"]),
                            use_container_width=True)


def render() -> None:
    st.title("Engine health check")
    st.markdown("Upload recent sensor readings to see which engines are likely to "
                "**fail within the next 30 cycles**, and why.")

    uploaded = st.file_uploader(
        "Sensor readings (CSV file)", type=["csv"],
        help="One row per engine per cycle, with columns unit, cycle and the sensor "
             "readings. See the Help page for the exact format and a template.")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        check = st.button("Check engines", type="primary", disabled=uploaded is None,
                          use_container_width=True)
    with col_b:
        if EXAMPLE_FILE.exists() and st.button("Try with example data"):
            run_prediction(EXAMPLE_FILE.name, EXAMPLE_FILE.read_bytes())
    if check and uploaded is not None:
        run_prediction(uploaded.name, uploaded.getvalue())

    if "prediction_error" in st.session_state:
        show_api_error(st.session_state["prediction_error"])
    elif "prediction" in st.session_state:
        st.divider()
        show_results(st.session_state["prediction"])
    else:
        st.info("No results yet. Upload a file and press **Check engines**, "
                "or press **Try with example data**.")
