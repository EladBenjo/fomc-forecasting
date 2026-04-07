"""Feature explorer page for Streamlit MVP."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.artifacts import load_features_dataframe, remediation_commands

st.title("Feature Explorer")
st.caption("Read-only summary views over `data/features/doc_level/features.parquet`.")

commands = remediation_commands()

try:
    features = load_features_dataframe()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.write("Build features with:")
    st.code(commands["features"])
    st.stop()

if features.empty:
    st.warning("Feature artifact exists but has no rows.")
    st.stop()

st.subheader("Dataset Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", int(len(features)))
with col2:
    st.metric("Documents", int(features["doc_id"].nunique()) if "doc_id" in features.columns else 0)
with col3:
    st.metric("Source Types", int(features["source_type"].nunique()) if "source_type" in features.columns else 0)

if "date" in features.columns:
    st.write(
        f"Date range: `{features['date'].min().date()}` -> `{features['date'].max().date()}`"
    )

if "source_type" in features.columns:
    st.subheader("Source Type Counts")
    source_counts = features["source_type"].value_counts().sort_index()
    st.bar_chart(source_counts)

if "date" in features.columns:
    monthly = (
        features.set_index("date")
        .resample("MS")
        .agg(
            hawkish_score=("hawkish_score", "mean") if "hawkish_score" in features.columns else ("doc_id", "count"),
            novelty=("novelty", "mean") if "novelty" in features.columns else ("doc_id", "count"),
            doc_count=("doc_id", "count") if "doc_id" in features.columns else ("source_type", "count"),
        )
        .reset_index()
    )
    st.subheader("Monthly Feature Trends")
    trend_cols = [col for col in ["hawkish_score", "novelty"] if col in monthly.columns]
    if trend_cols:
        st.line_chart(monthly.set_index("date")[trend_cols])
    if "doc_count" in monthly.columns:
        st.area_chart(monthly.set_index("date")[["doc_count"]])

st.subheader("Sample Rows")
st.dataframe(features.head(100))

