"""Fed Communication Monitor page for economist dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib.artifacts import remediation_commands
from app.lib.cached_data import get_features_frame, get_optional_metadata_frame
from app.lib.dashboard_metrics import monthly_communication_frame, recent_events_table

st.title("Fed Communication Monitor")
st.caption("Time-series monitoring of Communication Signal, Policy Tone, Novelty, and document flow.")

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
source_filter_options = ["all"] + source_values
selected_source = st.selectbox("Source filter", options=source_filter_options, index=0)

filtered = features.copy()
if selected_source != "all":
    filtered = filtered[filtered["source_type"] == selected_source].copy()

if filtered.empty:
    st.info("No rows match the current source filter.")
    st.stop()

monthly = monthly_communication_frame(filtered)
if monthly.empty:
    st.warning("Unable to compute monthly communication metrics from current selection.")
    st.stop()

latest_row = monthly.iloc[-1]
top_metrics = st.columns(4)
with top_metrics[0]:
    st.metric("Latest Feature Month", str(latest_row["feature_month"].date()))
with top_metrics[1]:
    if "hawkish_score_mean" in monthly.columns:
        st.metric("Policy Tone (Monthly Mean)", f"{float(latest_row['hawkish_score_mean']):.3f}")
with top_metrics[2]:
    if "novelty_mean" in monthly.columns:
        st.metric("Novelty (Monthly Mean)", f"{float(latest_row['novelty_mean']):.3f}")
with top_metrics[3]:
    st.metric("Document Count", int(latest_row["doc_count"]))

st.subheader("Communication Signal and Novelty Trends")
trend_cols = [col for col in ["hawkish_score_mean", "novelty_mean", "target_sentences_ratio_mean"] if col in monthly.columns]
if trend_cols:
    st.line_chart(monthly.set_index("feature_month")[trend_cols], use_container_width=True)
else:
    st.info("Required columns for trend chart are unavailable.")

st.subheader("Document Volume")
st.area_chart(monthly.set_index("feature_month")[["doc_count"]], use_container_width=True)

st.subheader("Text Length Distribution")
if "text_length_words" in filtered.columns and filtered["text_length_words"].notna().any():
    lengths = pd.to_numeric(filtered["text_length_words"], errors="coerce").dropna()
    bins = np.histogram_bin_edges(lengths, bins=20)
    binned = pd.cut(lengths, bins=bins, include_lowest=True).value_counts().sort_index()
    length_hist = pd.DataFrame({"bucket": binned.index.astype(str), "count": binned.values}).set_index("bucket")
    st.bar_chart(length_hist, use_container_width=True)
else:
    st.info("`text_length_words` is unavailable in the active feature source.")

st.subheader("Recent Event")
metadata = get_optional_metadata_frame()
recent_events = recent_events_table(filtered, metadata_df=metadata, lookback_days=60, top_n=1)
if recent_events.empty:
    st.info("No recent event-style signal found in the selected window.")
else:
    row = recent_events.iloc[0]
    st.write(
        f"Recent Event: `{row['date'].date()}` | {row['source_type']} `{row['title_or_doc']}` | "
        f"Communication Signal={float(row['hawkish_score']):.3f} | Novelty={float(row['novelty']):.3f}"
    )
    st.caption(row["explanation"])
