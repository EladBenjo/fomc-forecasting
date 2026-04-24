"""Feature driver diagnostics page for economist dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.artifacts import remediation_commands
from app.lib.cached_data import get_model_dataset_frame
from app.lib.dashboard_metrics import feature_correlation_table, recent_feature_zscores

st.title("Feature Drivers / Model Interpretation")
st.caption("Transparent diagnostics to assess which features may be associated with forecast behavior.")

commands = remediation_commands()
try:
    model_df = get_model_dataset_frame()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.code(commands["model_dataset"])
    st.stop()

if model_df.empty:
    st.warning("Model dataset exists but has no rows.")
    st.stop()

candidate_features = [
    "hawkish_score",
    "novelty",
    "doc_count",
    "text_length_words_max",
    "role_share_chairman",
    "hawkish_score_max_abs_signed_7d",
    "hawkish_score_max_abs_signed_14d",
    "hawkish_score_max_abs_signed_30d",
    "n_target_sentences",
    "n_hawkish",
    "n_dovish",
]
available_features = [col for col in candidate_features if col in model_df.columns]

st.subheader("Feature Correlations with Target")
corr_df = feature_correlation_table(
    model_df,
    target_col="t5yie_diff1",
    candidate_cols=available_features,
)
if corr_df.empty:
    st.info("No sufficient overlap to compute feature-target correlations.")
else:
    st.dataframe(corr_df, hide_index=True, use_container_width=True)

st.subheader("Feature vs Target Scatter")
if not available_features:
    st.info("No candidate features available in current dataset.")
else:
    default_feature = (
        corr_df.iloc[0]["feature"] if not corr_df.empty else available_features[0]
    )
    selected_feature = st.selectbox(
        "Feature",
        options=available_features,
        index=available_features.index(default_feature) if default_feature in available_features else 0,
    )
    scatter_df = model_df[[selected_feature, "t5yie_diff1"]].dropna().rename(
        columns={selected_feature: "feature_value", "t5yie_diff1": "target_value"}
    )
    if scatter_df.empty:
        st.info("No overlapping observations for selected feature.")
    else:
        st.scatter_chart(scatter_df, x="feature_value", y="target_value", use_container_width=True)

st.subheader("Recent Feature Movements")
top_features = corr_df["feature"].head(4).tolist() if not corr_df.empty else available_features[:4]
zscores = recent_feature_zscores(model_df, feature_cols=top_features, periods=24)
if zscores.empty:
    st.info("Unable to build recent movement panel.")
else:
    st.line_chart(zscores.set_index("date"), use_container_width=True)
    st.caption("Series are standardized (z-score) over the displayed window.")

st.subheader("Event-Style Sentiment vs Monthly Mean Sentiment")
event_col = "hawkish_score_max_abs_signed_30d"
mean_col = "hawkish_score"
if event_col in model_df.columns and mean_col in model_df.columns:
    comp = model_df[["date", event_col, mean_col]].dropna().sort_values("date", kind="mergesort")
    if comp.empty:
        st.info("No overlapping rows for event-style and monthly-mean sentiment.")
    else:
        st.line_chart(
            comp.set_index("date").rename(
                columns={
                    event_col: "Event-style 30d max-abs signed",
                    mean_col: "Monthly mean hawkish_score",
                }
            ),
            use_container_width=True,
        )
        corr = comp[event_col].corr(comp[mean_col], method="spearman")
        delta = comp[event_col] - comp[mean_col]
        st.metric("Spearman correlation", f"{float(corr):.3f}")
        st.metric("Mean level difference (event - mean)", f"{float(delta.mean()):.3f}")
else:
    st.info("Event-style sentiment columns are unavailable in this dataset.")

st.subheader("Limitations")
st.write(
    "- Correlation diagnostics are associative, not causal.\n"
    "- Feature relevance can change across policy regimes.\n"
    "- Directional and level metrics should be interpreted jointly."
)
