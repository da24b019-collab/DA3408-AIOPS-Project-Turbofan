"""
"Monitoring" page: is the incoming data still like the training data?

    - Data drift state and the PSI of each sensor (from the backend's /drift/status)
    - Alerts that Alertmanager has sent to the backend
    - A link to Grafana, where the live graphs are

The full time history lives in Prometheus and Grafana; this page shows the
current state in words an operator can act on.
"""

import os

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError
from ui import show_api_error

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000")

STATE_TEXT = {
    "no_baseline": ("info", "The current model has no drift baseline (it was trained before "
                            "monitoring was added). Retrain it to enable drift checks."),
    "collecting": ("info", "Collecting readings: the drift check starts after {min_rows:,} rows "
                           "({n_rows:,} received so far). Fewer rows give unreliable results."),
    "ok": ("success", "No drift: recent sensor readings look like the training data "
                      "(highest PSI {max_psi:.2f}, alarm above {threshold:.1f})."),
    "drift": ("error", "Data drift detected: sensor readings no longer look like the training "
                       "data (highest PSI {max_psi:.2f} on {worst_sensor}, alarm above "
                       "{threshold:.1f}). Predictions may be unreliable; consider retraining."),
}


def drift_message(status: dict) -> tuple:
    """(kind, text) for the drift state, e.g. ("error", "Data drift detected: ...")."""
    kind, template = STATE_TEXT[status["state"]]
    return kind, template.format(**status)


def psi_table(status: dict) -> pd.DataFrame:
    """One row per sensor, highest PSI first, with a plain-language verdict."""
    rows = [{"Sensor": sensor, "PSI": value,
             "Verdict": "drifted" if value > status["threshold"] else "stable"}
            for sensor, value in status["sensor_psi"].items()]
    return pd.DataFrame(rows).sort_values("PSI", ascending=False, ignore_index=True)


def jobs_table(job_list: list) -> pd.DataFrame:
    """The columns an operator needs, newest job first."""
    rows = [{"Job": j["id"], "Kind": j["kind"], "Status": j["status"], "Reason": j["reason"],
             "Created": j["created_at"].replace("T", " ")[:19],
             "Result": j["detail"] or "-"} for j in job_list]
    return pd.DataFrame(rows)


def render() -> None:
    st.title("Monitoring")
    st.markdown("Checks that the sensor data arriving now still looks like the data the "
                "model learned from. If it does not, the predictions cannot be trusted.")
    client = ApiClient()
    c1, c2 = st.columns([1, 5])
    if c1.button("Refresh"):
        st.rerun()

    # ---- data drift ----
    st.subheader("Data drift")
    try:
        status = client.drift_status()
        kind, text = drift_message(status)
        getattr(st, kind)(text)
        m1, m2, m3 = st.columns(3)
        m1.metric("Readings in window", f"{status['n_rows']:,} / {status['window_rows']:,}")
        m2.metric("Highest PSI", "-" if status["max_psi"] is None else f"{status['max_psi']:.2f}")
        m3.metric("Baseline from model version", status["baseline_model_version"] or "-")
        if status["sensor_psi"]:
            table = psi_table(status)
            st.bar_chart(table, x="Sensor", y="PSI", horizontal=True, sort="-PSI")
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"PSI": st.column_config.NumberColumn(format="%.3f")})
        st.caption("PSI (Population Stability Index) compares how each sensor's recent "
                   "readings are spread out with its training readings. 0 = identical.")
        if c2.button("Empty the drift window",
                     help="Start counting again, e.g. after you have dealt with a drift alert."):
            client.reset_drift()
            st.rerun()
    except ApiError as err:
        show_api_error(err)

    # ---- alerts ----
    st.subheader("Recent alerts")
    try:
        alerts = client.recent_alerts(limit=20)["alerts"]
        if alerts:
            st.dataframe(pd.DataFrame(alerts)[["received_at", "alertname", "status",
                                               "severity", "summary", "action"]],
                         hide_index=True, width="stretch")
        else:
            st.success("No alerts received since the backend started.")
    except ApiError as err:
        show_api_error(err)

    # ---- retraining ----
    st.subheader("Retraining jobs")
    st.caption("A DataDrift alert queues a retraining job automatically. The retraining worker "
               "on the server trains a new model, which then waits for approval on the Model page.")
    try:
        job_list = client.jobs(limit=10)["jobs"]
        if job_list:
            st.dataframe(jobs_table(job_list), hide_index=True, width="stretch")
        else:
            st.info("No retraining or deploy jobs yet.")
    except ApiError as err:
        show_api_error(err)
    with st.expander("Start a retraining job by hand"):
        subsets = st.multiselect("Train on", ["FD001", "FD002", "FD003", "FD004"],
                                 default=["FD001", "FD002"])
        if st.button("Queue retraining job", disabled=not subsets):
            try:
                job = client.retrain(subsets, "Requested from the dashboard")["jobs"][0]
                st.success(f"Retrain job {job['id']} queued.")
            except ApiError as err:
                show_api_error(err)

    st.link_button("Open the live graphs in Grafana", GRAFANA_URL)
