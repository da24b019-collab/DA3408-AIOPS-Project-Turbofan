"""
"Model" page: which model is making the predictions, and how good it is.

Since Task 9 also the human approval step: compare the newly trained
"challenger" with the current "champion" on the same test engines, approve
it, or roll back to an earlier version.
"""

import os

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError
from ui import show_api_error

MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")


SUBSET_MEANING = {
    "FD001": "1 flight condition, 1 fault (original task)",
    "FD002": "6 flight conditions, 1 fault",
    "FD003": "1 flight condition, 2 faults",
    "FD004": "6 flight conditions, 2 faults",
}


def comparison_table(result: dict) -> pd.DataFrame:
    """One row per test set: recall and PR-AUC of both models."""
    rows = []
    for subset, meaning in SUBSET_MEANING.items():
        old = result["scores"]["champion"][subset]
        new = result["scores"]["challenger"][subset]
        rows.append(
            {
                "Test engines": f"{subset}: {meaning}",
                "Recall (champion)": old["recall"],
                "Recall (challenger)": new["recall"],
                "PR-AUC (champion)": old["pr_auc"],
                "PR-AUC (challenger)": new["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def render_approval(client: ApiClient) -> None:
    st.subheader("New model waiting for approval")
    try:
        result = client.compare_models()
    except ApiError as err:
        show_api_error(err)
        return
    champion, challenger = result["champion_version"], result["challenger_version"]
    if champion == challenger:
        st.info(
            f"No new model: the latest trained version ({challenger}) is already in use. "
            "A retraining job (Monitoring page) creates a new one."
        )
        return

    st.markdown(
        f"Version **{challenger}** (challenger) was trained and is waiting. "
        f"Both models were scored on the same NASA test engines. Recall = share of "
        f"failing engines caught, with each model's own threshold."
    )
    st.dataframe(
        comparison_table(result),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in comparison_table(result).columns[1:]
        },
    )
    for check in result["checks"]:
        icon = ":material/check_circle:" if check["passed"] else ":material/cancel:"
        st.markdown(f"{icon} {check['name']} ({check['detail']})")

    if not result["approvable"]:
        st.error("Approval is blocked: the new model is clearly worse on the original task.")
        return
    reviewed = st.checkbox(
        f"I have reviewed version {challenger} and want it to replace " f"version {champion}"
    )
    if st.button(f"Approve version {challenger}", type="primary", disabled=not reviewed):
        try:
            answer = client.approve()
            st.success(
                f"Approved. Version {answer['champion_version']} is now champion. Deploy "
                f"job {answer['job']['id']} restarts the model server (about 30 s); "
                "follow it on the Monitoring page."
            )
        except ApiError as err:
            show_api_error(err)


def render_rollback(client: ApiClient, current: str) -> None:
    st.subheader("Roll back")
    st.caption(
        "If the current model misbehaves, put an earlier version back in use. "
        "No checks are applied: in an emergency, speed matters."
    )
    try:
        versions = client.model_versions()["versions"]
    except ApiError as err:
        show_api_error(err)
        return
    earlier = [v for v in versions if v["version"] != current]
    if not earlier:
        st.info("There is no other version to roll back to.")
        return
    labels = {
        f"Version {v['version']} (trained on {v['train_subsets']}, "
        f"test recall {v['test_recall'] or '-'})": v["version"]
        for v in earlier
    }
    choice = st.selectbox("Version to put back", list(labels))
    if st.button("Roll back to this version"):
        try:
            answer = client.rollback(labels[choice])
            st.success(
                f"Version {answer['champion_version']} is champion again. Deploy job "
                f"{answer['job']['id']} restarts the model server."
            )
        except ApiError as err:
            show_api_error(err)


def render() -> None:
    st.title("Current model")
    try:
        info = ApiClient().model_info()
    except ApiError as err:
        show_api_error(err)
        return

    st.markdown(
        f"The dashboard uses version **{info['version']}** of "
        f"`{info['model_name']}`, labelled **{info['alias']}** in the model registry."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model type", info["model_type"])
    c2.metric(
        "Decision threshold",
        f"{info['threshold']:.3f}",
        help="Engines with a failure probability at or above this are high risk.",
    )
    c3.metric(
        "Test recall",
        f"{info['test_metrics']['recall']:.2f}",
        help="Share of truly at-risk test engines the model caught.",
    )
    c4.metric(
        "Test PR-AUC",
        f"{info['test_metrics']['pr_auc']:.3f}",
        help="Overall ranking quality for rare failures (1.0 is perfect).",
    )

    st.subheader("Where it came from")
    st.markdown(
        f"- **Training run:** `{info['run_id']}`\n"
        f"- **Code version (Git commit):** `{info['git_commit'][:12]}`\n"
        f"- **Loaded by the backend at:** {info['loaded_at'].replace('T', ' ')}\n"
        f"- **Looks at:** the last {info['window']} cycles of each engine "
        f"({info['n_features']} values computed from {len(info['sensors'])} sensors)"
    )
    st.link_button("Open the model registry in MLflow", f"{MLFLOW_UI_URL}/#/models")

    st.subheader("Sensors the model uses")
    st.dataframe(
        pd.DataFrame(info["sensors"]).rename(columns={"code": "Column", "description": "Sensor"}),
        hide_index=True,
        use_container_width=True,
    )

    client = ApiClient()
    render_approval(client)
    render_rollback(client, info["version"])
