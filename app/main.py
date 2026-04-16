"""Streamlit entrypoint for the Phase 7 MVP app."""

from __future__ import annotations

import streamlit as st

from app.lib.artifacts import build_status_snapshot

st.set_page_config(page_title="FOMC Forecasting", layout="wide")

st.title("FOMC Forecasting App (MVP)")
st.caption("Read-only app wiring existing pipeline artifacts into interactive views.")
st.write(
    "Use the sidebar to navigate: **Pipeline Status**, **Feature Explorer**, "
    "**Model Results**, and **RAG Chat** (placeholder)."
)

snapshot = build_status_snapshot()
checks = snapshot["checks"]

cols = st.columns(len(checks))
for col, check in zip(cols, checks):
    with col:
        status_text = "Ready" if check["ready"] else "Missing"
        st.metric(check["label"], status_text)
        st.caption(check["path"])

if snapshot["ready_count"] < snapshot["total_checks"]:
    st.info("Some artifacts are missing. Generate them with the commands below.")
    for check in checks:
        if check["ready"]:
            continue
        st.write(f"**{check['label']}**")
        st.code(check["remediation_command"])
else:
    st.success("All core MVP artifacts are available.")

latest_run = snapshot["runs"]["latest_complete"]
if latest_run:
    st.write(f"Latest complete Phase 4 run: `{latest_run}`")

