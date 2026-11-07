"""
Shared look and feel: risk colours, badges and a little extra styling.

Risk is always shown with BOTH a colour and a word ("High"), so the
dashboard still works for colour-blind users and on black-and-white prints.
"""

import streamlit as st

RISK_COLOURS = {
    "high": {"text": "#9B1C1C", "background": "#FDE8E8", "label": "High"},
    "medium": {"text": "#8A5A00", "background": "#FEF3C7", "label": "Medium"},
    "low": {"text": "#1E6B3A", "background": "#DEF7E5", "label": "Low"},
}

CUSTOM_CSS = """
<style>
/* Cards around the summary numbers */
div[data-testid="stMetric"] {
    background: #F2F5F9;
    border: 1px solid #E1E7EF;
    border-radius: 10px;
    padding: 12px 16px;
}
/* Coloured risk labels used in text */
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
}
.muted { color: #5B6B7D; font-size: 0.9rem; }
</style>
"""


def apply_style() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    """HTML for a coloured pill such as [ High ]."""
    c = RISK_COLOURS[level]
    return (
        f'<span class="risk-badge" style="color:{c["text"]};'
        f'background:{c["background"]}">{c["label"]}</span>'
    )


def risk_label(level: str) -> str:
    """Plain-text version for tables: a coloured dot plus the word."""
    dot = {"high": "🔴", "medium": "🟠", "low": "🟢"}[level]
    return f"{dot} {RISK_COLOURS[level]['label']}"


def show_api_error(err) -> None:
    """Show a backend error with a hint about what to do next."""
    hints = {
        "invalid_input": "Check that the file has the columns listed on the Help page.",
        "bad_file": "Upload a .csv file (comma-separated values) smaller than 5 MB.",
        "model_unavailable": "The model is not available right now. Try again in a minute, "
        "or ask the system administrator.",
        "service_unavailable": "A supporting service is down. Try again in a minute.",
        "connection_error": "The prediction service is not running or not reachable.",
        "unauthorized": "The dashboard's API key does not match the backend's.",
        "conflict": "This action is not possible right now (see the message above).",
        "not_found": "That item does not exist. Refresh the page.",
    }
    st.error(str(err))
    if err.code in hints:
        st.caption(f"What to do: {hints[err.code]}")
