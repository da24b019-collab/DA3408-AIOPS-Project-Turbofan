"""
Turbofan engine health dashboard (Streamlit).

This is a separate program from the backend: it only talks to the backend's
REST API, at the address in the BACKEND_URL environment variable.

Run it from the frontend folder (so Streamlit finds .streamlit/config.toml):
    cd frontend
    streamlit run app.py
Then open http://localhost:8501
"""

import streamlit as st
from api_client import BACKEND_URL, ApiClient, ApiError
from ui import apply_style
from views import help as help_page
from views import model, monitoring, pipeline, predict

st.set_page_config(page_title="Engine Health Dashboard", page_icon="🛠️", layout="wide")
apply_style()


@st.cache_data(ttl=15, show_spinner=False)
def backend_status() -> dict:
    """Ask the backend if it is ready. Cached for 15 s so every click does not call it."""
    try:
        return ApiClient().ready()
    except ApiError:
        return {"ready": False, "model_loaded": False, "unreachable": True}


def sidebar_status() -> None:
    """Small status box on every page, so problems are visible straight away."""
    status = backend_status()
    with st.sidebar:
        st.markdown("### System status")
        if status.get("unreachable"):
            st.error("Prediction service unreachable")
            st.caption(f"Expected at {BACKEND_URL}")
        elif status.get("ready"):
            st.success(f"Ready · model version {status.get('model_version')}")
        else:
            st.warning("Starting up or model not available")
            if status.get("model_server_in_sync") is False:
                st.caption("The model server is running a different model version.")
        st.caption("Status refreshes every 15 seconds.")


pages = st.navigation(
    [
        st.Page(
            predict.render, title="Engine check", icon=":material/monitor_heart:", default=True
        ),
        st.Page(
            pipeline.render, title="Pipeline", icon=":material/account_tree:", url_path="pipeline"
        ),
        st.Page(
            monitoring.render,
            title="Monitoring",
            icon=":material/monitoring:",
            url_path="monitoring",
        ),
        st.Page(model.render, title="Model", icon=":material/model_training:", url_path="model"),
        st.Page(help_page.render, title="Help", icon=":material/help:", url_path="help"),
    ]
)
sidebar_status()
pages.run()
