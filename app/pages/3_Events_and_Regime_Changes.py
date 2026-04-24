"""Events and regime-change page for economist dashboard."""

from __future__ import annotations

import streamlit as st

from app.lib.artifacts import remediation_commands
from app.lib.cached_data import get_features_frame, get_optional_metadata_frame
from app.lib.dashboard_metrics import detect_regime_markers, monthly_communication_frame, recent_events_table

st.title("Events & Regime Changes")
st.caption("Diagnostics for Novelty Spike periods, extreme policy tone, and potential communication regime shifts.")

commands = remediation_commands()
try:
    features = get_features_frame(prefer_backfilled=True)
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.code(commands["features"])
    st.stop()

if features.empty:
    st.warning("Feature artifact exists but has no rows.")
    st.stop()

source_values = sorted(features["source_type"].dropna().unique().tolist()) if "source_type" in features.columns else []
selected_sources = st.multiselect(
    "Source types",
    options=source_values,
    default=source_values,
)
if selected_sources:
    filtered = features[features["source_type"].isin(selected_sources)].copy()
else:
    filtered = features.copy()

if filtered.empty:
    st.info("No rows match the selected source filters.")
    st.stop()

monthly = monthly_communication_frame(filtered)
if monthly.empty:
    st.warning("Unable to compute monthly diagnostics from current selection.")
    st.stop()

st.subheader("Monthly Signal Context")
chart_cols = [col for col in ["hawkish_score_mean", "novelty_mean"] if col in monthly.columns]
if chart_cols:
    st.line_chart(monthly.set_index("feature_month")[chart_cols], use_container_width=True)
else:
    st.info("Hawkish/novelty columns are unavailable in current data.")

markers = detect_regime_markers(monthly)
st.subheader("Regime-Change Heuristic Markers")
if markers.empty:
    st.info("No heuristic regime markers triggered with current thresholds.")
else:
    marker_counts = markers["marker_type"].value_counts().to_dict()
    marker_cols = st.columns(3)
    marker_cols[0].metric("Novelty Spike markers", int(marker_counts.get("Novelty Spike", 0)))
    marker_cols[1].metric("Policy Tone Shift markers", int(marker_counts.get("Policy Tone Shift", 0)))
    marker_cols[2].metric("Total markers", int(len(markers)))
    st.dataframe(markers, hide_index=True, use_container_width=True)
    st.caption("Markers are transparent heuristics, not causal regime labels.")

st.subheader("Recent Events")
lookback_days = st.slider("Recent event lookback (days)", min_value=30, max_value=730, value=180, step=30)
top_n = st.slider("Rows", min_value=10, max_value=100, value=30, step=10)
metadata = get_optional_metadata_frame()
events = recent_events_table(
    filtered,
    metadata_df=metadata,
    lookback_days=lookback_days,
    top_n=top_n,
)
if events.empty:
    st.info("No events found in selected window.")
else:
    st.dataframe(events, hide_index=True, use_container_width=True)
