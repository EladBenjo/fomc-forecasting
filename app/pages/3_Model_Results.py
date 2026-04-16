"""Model results page for Streamlit MVP."""

from __future__ import annotations

import streamlit as st

from app.lib.artifacts import (
    latest_phase4_run,
    list_phase4_runs,
    load_phase4_run_artifacts,
    remediation_commands,
)

st.title("Model Results")
st.caption("Phase 4 SARIMAX baseline vs exogenous benchmark artifacts.")

commands = remediation_commands()
runs = list_phase4_runs(require_complete=True)

if not runs:
    st.warning("No complete Phase 4 run artifacts found.")
    st.write("Generate a run with:")
    st.code(commands["phase4"])
    st.stop()

latest = latest_phase4_run(require_complete=True)
default_index = runs.index(latest) if latest in runs else len(runs) - 1
selected_run = st.selectbox("Run version", options=runs, index=default_index)

try:
    payload = load_phase4_run_artifacts(run_version=selected_run)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code(commands["phase4"])
    st.stop()

results = payload["results_table"]
paired = payload["paired_comparison"]
summary = payload["run_summary"]
config = payload["run_config"]
predictions = payload["predictions"]

st.subheader("Run Summary")
st.json(summary)

st.subheader("Results Table")
st.dataframe(results)

st.subheader("Paired Comparison")
st.dataframe(paired)

st.subheader("Predictions")
if predictions.empty:
    st.warning("Predictions artifact is empty.")
else:
    split_options = sorted(predictions["split"].dropna().unique().tolist())
    split_selected = st.selectbox("Split", split_options, index=0)

    split_df = predictions[predictions["split"] == split_selected]
    variant_options = sorted(split_df["model_variant"].dropna().unique().tolist())
    variant_selected = st.selectbox("Model variant", variant_options, index=0)

    view = (
        split_df[split_df["model_variant"] == variant_selected][["date", "actual", "pred"]]
        .sort_values("date")
        .set_index("date")
    )
    st.line_chart(view)
    st.dataframe(view.reset_index())

with st.expander("Run Config"):
    st.json(config)

