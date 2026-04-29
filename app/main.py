"""Streamlit entrypoint for the economist dashboard."""

from __future__ import annotations

import streamlit as st

from app.lib.cached_data import (
    get_artifact_freshness,
    get_latest_phase4_payload,
    get_latest_xgboost_payload,
    get_model_dataset_frame,
    get_status_snapshot,
)
from app.lib.dashboard_metrics import classify_policy_tone

st.set_page_config(page_title="Fed Communication Dashboard", layout="wide")


def _render_latest_forecast(label: str, payload, remediation_command: str) -> None:
    st.caption(label)
    if payload is None:
        st.warning("Forecast artifacts unavailable.")
        st.code(remediation_command)
        return

    preds = payload["predictions"]
    if preds.empty:
        st.warning("Predictions artifact exists but is empty.")
        return

    latest_pred = preds.sort_values("date", kind="mergesort").iloc[-1]
    st.metric("Latest Forecast Date", str(latest_pred["date"].date()))
    st.metric("Latest Model Prediction", f"{float(latest_pred['pred']):.3f}")
    st.caption(f"Variant: `{latest_pred['model_variant']}`")
    st.caption(f"Run: `{payload['run_version']}`")


st.title("Federal Reserve Communication Dashboard")
st.caption("Business Analyst / Economist workspace for communication signals and inflation-expectations forecasting.")
st.write(
    "Use the sidebar to navigate Executive Snapshot, Communication Monitor, Event/Regime diagnostics, "
    "Forecast Results, and Feature Drivers."
)

snapshot = get_status_snapshot()
checks = snapshot["checks"]

cols = st.columns(len(checks))
for col, check in zip(cols, checks):
    with col:
        status_text = "Ready" if check["ready"] else "Missing"
        st.metric(check["label"], status_text)
if snapshot.get("selected_feature_path"):
    st.caption(f"Communication feature source: `{snapshot['selected_feature_path']}`")

if snapshot["ready_count"] < snapshot["total_checks"]:
    st.info("Some artifacts are missing. Generate artifacts with the commands below.")
    for check in checks:
        if check["ready"]:
            continue
        st.write(f"**{check['label']}**")
        st.code(check["remediation_command"])
else:
    st.success("Core dashboard artifacts are available.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Current Signal Snapshot")
    try:
        model_df = get_model_dataset_frame()
    except FileNotFoundError:
        st.warning("Model dataset is unavailable. Build with `python -m datasets.build_dataset.builder`.")
    else:
        if model_df.empty:
            st.warning("Model dataset is available but empty.")
        else:
            latest = model_df.sort_values("date", kind="mergesort").iloc[-1]
            prev = model_df.sort_values("date", kind="mergesort").iloc[-2] if len(model_df) > 1 else None
            latest_target = latest.get("t5yie_diff1")
            prev_target = prev.get("t5yie_diff1") if prev is not None else None
            delta = (
                float(latest_target - prev_target)
                if prev_target is not None and latest_target is not None
                else None
            )
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Latest Target Month", str(latest["date"].date()))
                if latest_target is not None:
                    st.metric(
                        "Latest Inflation-Expectation Change",
                        f"{float(latest_target):.3f}",
                        None if delta is None else f"{delta:+.3f}",
                    )
            with m2:
                if "hawkish_score" in model_df.columns:
                    tone_score = float(model_df["hawkish_score"].iloc[-1])
                    st.metric("Policy Tone", classify_policy_tone(tone_score))
                    st.caption(f"Communication Signal: {tone_score:.3f}")
                if "novelty" in model_df.columns:
                    st.metric("Latest Novelty", f"{float(model_df['novelty'].iloc[-1]):.3f}")

with right:
    st.subheader("Latest Forecast Availability")
    forecast_cols = st.columns(2)
    with forecast_cols[0]:
        _render_latest_forecast("SARIMAX", get_latest_phase4_payload(), "python -m models.baselines.sarimax")
    with forecast_cols[1]:
        _render_latest_forecast("XGBoost", get_latest_xgboost_payload(), "python -m models.ml.xgboost")

st.subheader("Data Freshness")
freshness = get_artifact_freshness()
st.dataframe(freshness, use_container_width=True, hide_index=True)

