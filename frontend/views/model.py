"""
"Model" page: which model is making the predictions, and how good it is.
"""

import os

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError
from ui import show_api_error

MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")


def render() -> None:
    st.title("Current model")
    try:
        info = ApiClient().model_info()
    except ApiError as err:
        show_api_error(err)
        return

    st.markdown(f"The dashboard uses version **{info['version']}** of "
                f"`{info['model_name']}`, labelled **{info['alias']}** in the model registry.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model type", info["model_type"])
    c2.metric("Decision threshold", f"{info['threshold']:.3f}",
              help="Engines with a failure probability at or above this are high risk.")
    c3.metric("Test recall", f"{info['test_metrics']['recall']:.2f}",
              help="Share of truly at-risk test engines the model caught.")
    c4.metric("Test PR-AUC", f"{info['test_metrics']['pr_auc']:.3f}",
              help="Overall ranking quality for rare failures (1.0 is perfect).")

    st.subheader("Where it came from")
    st.markdown(
        f"- **Training run:** `{info['run_id']}`\n"
        f"- **Code version (Git commit):** `{info['git_commit'][:12]}`\n"
        f"- **Loaded by the backend at:** {info['loaded_at'].replace('T', ' ')}\n"
        f"- **Looks at:** the last {info['window']} cycles of each engine "
        f"({info['n_features']} values computed from {len(info['sensors'])} sensors)")
    st.link_button("Open the model registry in MLflow", f"{MLFLOW_UI_URL}/#/models")

    st.subheader("Sensors the model uses")
    st.dataframe(pd.DataFrame(info["sensors"]).rename(
        columns={"code": "Column", "description": "Sensor"}),
        hide_index=True, use_container_width=True)
