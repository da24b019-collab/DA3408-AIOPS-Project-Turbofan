"""
"Pipeline" page: the ML pipeline at a glance.

    - The DVC pipeline drawn as a diagram, each stage coloured by its status
    - Speed and throughput of each stage (rows, seconds, rows per second)
    - The history of training runs from MLflow, successful and failed
    - Recent warnings and errors from the logs
    - Links to each tool's own interface (MLflow, Grafana, Spark)
"""

import os

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError
from ui import show_api_error

TOOL_LINKS = {
    "MLflow (experiments and model registry)": os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000"),
    "Grafana (live monitoring)": os.getenv("GRAFANA_URL", "http://127.0.0.1:3000"),
    "Spark UI (only while a Spark job runs)": os.getenv("SPARK_UI_URL", "http://127.0.0.1:4040"),
}
STATUS_STYLE = {  # fill colour and word for each stage status
    "up_to_date": ("#DEF7E5", "up to date"),
    "changed": ("#FEF3C7", "needs re-run"),
    "unknown": ("#E5E7EB", "status unknown"),
}


def dag_dot(status: dict) -> str:
    """Graphviz description of the pipeline: one box per stage, arrows between them."""
    lines = ['digraph pipeline {', 'rankdir=LR;',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             '"data/raw.dvc" [label="Raw NASA data\\n(DVC tracked)", fillcolor="#F2F5F9"];']
    for stage in status["stages"]:
        fill, word = STATUS_STYLE[stage["status"]]
        lines.append(f'"{stage["name"]}" [label="{stage["name"]}\\n({word})", fillcolor="{fill}"];')
    for source, target in status["edges"]:
        lines.append(f'"{source}" -> "{target}";')
    lines.append("}")
    return "\n".join(lines)


def stage_cards(stages: list) -> None:
    """One card per stage with its key numbers."""
    columns = st.columns(len(stages))
    for col, stage in zip(columns, stages):
        m = stage["metrics"]
        with col:
            st.markdown(f"**{stage['name']}**")
            if "rows_per_second" in m:
                st.metric("Throughput", f"{m['rows_per_second']:,} rows/s")
                st.caption(f"{m['total_rows']:,} rows in {m['processing_seconds']} s "
                           f"(+ {m.get('spark_startup_seconds', 0)} s Spark start-up)")
            elif "total_rows" in m:
                st.metric("Rows ingested", f"{m['total_rows']:,}")
                st.caption(f"{m.get('engines', 0):,} engines across all subsets")
            elif "test_recall" in m:
                st.metric("Test recall", f"{m['test_recall']:.2f}")
                st.caption(f"{m.get('model_type', '')} · version {m.get('model_version', '?')} "
                           f"· trained in {m.get('train_seconds', '?')} s")
            last = (stage["last_run"] or "never")[:16].replace("T", " ")
            st.caption(f"Last run: {last}")


def runs_table(runs: list) -> pd.DataFrame:
    icon = {"FINISHED": "✅ finished", "FAILED": "❌ failed",
            "RUNNING": "⏳ running", "KILLED": "⛔ stopped"}
    return pd.DataFrame([{
        "Started": r["start_time"][:16].replace("T", " "),
        "Run": r["run_name"],
        "Status": icon.get(r["status"], r["status"]),
        "Model version": r["model_version"] or "-",
        "Label": ", ".join(r["aliases"]) or "",
        "Duration (s)": r["duration_seconds"],
        "Test recall": r["test_recall"],
        "Test PR-AUC": r["test_pr_auc"],
        "Git commit": r["git_commit"],
    } for r in runs])


def render() -> None:
    st.title("Pipeline status")
    st.markdown("How the data flows from raw files to the deployed model, how fast each "
                "step runs, and the history of training runs.")
    if st.button("Refresh"):
        st.rerun()
    client = ApiClient()

    # ---- DVC pipeline ----
    st.subheader("Data and training pipeline (DVC)")
    try:
        status = client.pipeline_status()
        st.graphviz_chart(dag_dot(status), use_container_width=True)
        if not status["dvc_available"]:
            st.caption("DVC is not available to the backend, so stage status is unknown.")
        st.caption("Green = up to date. Amber = its inputs changed since the last run "
                   "(run `dvc repro` to update).")
        stage_cards(status["stages"])
    except ApiError as err:
        show_api_error(err)

    # ---- MLflow runs ----
    st.subheader("Training runs (MLflow)")
    try:
        runs = client.pipeline_runs(limit=30)["runs"]
        if runs:
            finished = sum(r["status"] == "FINISHED" for r in runs)
            failed = sum(r["status"] == "FAILED" for r in runs)
            c1, c2, c3 = st.columns(3)
            c1.metric("Runs shown", len(runs))
            c2.metric("Successful", finished)
            c3.metric("Failed", failed)
            st.dataframe(runs_table(runs), hide_index=True, use_container_width=True,
                         column_config={"Label": st.column_config.TextColumn(width="medium"),
                                        "Test PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                        "Test recall": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.info("No training runs recorded yet.")
    except ApiError as err:
        show_api_error(err)

    # ---- warnings and errors ----
    st.subheader("Recent warnings and errors")
    try:
        events = client.recent_logs(limit=30)["events"]
        if events:
            st.dataframe(pd.DataFrame(events)[["time", "level", "source", "message", "file"]],
                         hide_index=True, use_container_width=True)
        else:
            st.success("No warnings or errors in the logs.")
    except ApiError as err:
        show_api_error(err)

    # ---- tool links ----
    st.subheader("Open the tools")
    for label, url in TOOL_LINKS.items():
        st.link_button(label, url)
