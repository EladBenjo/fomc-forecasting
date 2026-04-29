"""Forecast and model-results page for economist dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.artifacts import remediation_commands
from app.lib.cached_data import (
    get_phase4_payload,
    get_phase4_runs,
    get_xgboost_payload,
    get_xgboost_runs,
)
from app.lib.dashboard_metrics import directional_accuracy

st.title("Forecast & Model Results")
st.caption("Model performance, Forecast Error diagnostics, and model artifact inspection.")

commands = remediation_commands()
model_runs = {
    "SARIMAX": get_phase4_runs(require_complete=True),
    "XGBoost": get_xgboost_runs(require_complete=True),
}
available_models = [model for model, runs in model_runs.items() if runs]
if not available_models:
    st.warning("No complete SARIMAX or XGBoost model run artifacts found.")
    st.code(commands["phase4"])
    st.code(commands["xgboost"])
    st.stop()

selected_model = st.selectbox("Model family", options=available_models, index=0)
runs = model_runs[selected_model]
selected_run = st.selectbox("Run version", options=runs, index=len(runs) - 1)
try:
    if selected_model == "XGBoost":
        payload = get_xgboost_payload(selected_run)
    else:
        payload = get_phase4_payload(selected_run)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code(commands["xgboost"] if selected_model == "XGBoost" else commands["phase4"])
    st.stop()

predictions = payload["predictions"]
results = payload["results_table"].copy()
paired = payload.get("paired_comparison", pd.DataFrame()).copy()
summary = payload["run_summary"]
feature_importance = payload.get("feature_importance", pd.DataFrame()).copy()
feature_schema = payload.get("feature_schema", {})

st.subheader("Run Summary")
st.json(summary)

st.subheader("Performance Snapshot")
if not results.empty and {"split", "model_variant", "rmse", "mae"}.issubset(results.columns):
    test_rows = results[results["split"] == "test"].copy()
    best_test = test_rows.sort_values("rmse", kind="mergesort").head(1)
    baseline_test = test_rows[test_rows["model_variant"] == "baseline_univariate"].head(1)

    cols = st.columns(3)
    if not best_test.empty:
        best = best_test.iloc[0]
        cols[0].metric("Best Test Variant", str(best["model_variant"]))
        cols[1].metric("Best Test RMSE", f"{float(best['rmse']):.4f}")
        cols[2].metric("Best Test MAE", f"{float(best['mae']):.4f}")
    if not baseline_test.empty:
        base = baseline_test.iloc[0]
        st.caption(
            f"Baseline test metrics: RMSE={float(base['rmse']):.4f}, MAE={float(base['mae']):.4f}"
        )
else:
    st.info("Results table does not expose the expected RMSE/MAE schema.")

if not paired.empty:
    st.subheader("Baseline vs Text-Enhanced Model")
    st.dataframe(paired, hide_index=True, use_container_width=True)

if not feature_importance.empty and {"feature", "importance"}.issubset(feature_importance.columns):
    st.subheader("XGBoost Feature Importance")
    importance_df = (
        feature_importance.copy()
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    top_importance = importance_df.head(20)
    st.bar_chart(top_importance.set_index("feature")["importance"], use_container_width=True)
    st.dataframe(top_importance, hide_index=True, use_container_width=True)

if isinstance(feature_schema, dict) and feature_schema:
    with st.expander("XGBoost Feature Schema"):
        st.write(f"Target: `{feature_schema.get('target_column', 'unknown')}`")
        st.write(f"Feature count: `{len(feature_schema.get('feature_columns', []))}`")
        missing_rate = feature_schema.get("missing_rate", {})
        if isinstance(missing_rate, dict) and missing_rate:
            missing_df = (
                pd.DataFrame(
                    [{"feature": feature, "missing_rate": rate} for feature, rate in missing_rate.items()]
                )
                .sort_values("missing_rate", ascending=False, kind="mergesort")
                .reset_index(drop=True)
            )
            st.dataframe(missing_df, hide_index=True, use_container_width=True)

st.subheader("Actual vs Predicted")
if predictions.empty:
    st.warning("Predictions artifact is empty.")
    st.stop()

split_options = sorted(predictions["split"].dropna().unique().tolist())
selected_split = st.selectbox("Split", options=split_options, index=0)
split_df = predictions[predictions["split"] == selected_split].copy()

variant_options = sorted(split_df["model_variant"].dropna().unique().tolist())
default_variants: list[str] = ["baseline_univariate"] if "baseline_univariate" in variant_options else []
text_variant = "exog_text_event_role_variant"
if text_variant in variant_options:
    default_variants.append(text_variant)
if not default_variants and variant_options:
    default_variants = [variant_options[0]]

selected_variants = st.multiselect(
    "Model variants",
    options=variant_options,
    default=default_variants,
)
if not selected_variants:
    st.info("Select at least one model variant.")
    st.stop()

actual_series = (
    split_df[["date", "actual"]]
    .dropna(subset=["date"])
    .drop_duplicates(subset=["date"], keep="last")
    .sort_values("date", kind="mergesort")
    .set_index("date")
    .rename(columns={"actual": "Actual"})
)
plot_df = actual_series.copy()
error_df = pd.DataFrame(index=actual_series.index)

for variant in selected_variants:
    pred_series = (
        split_df[split_df["model_variant"] == variant][["date", "pred"]]
        .dropna(subset=["date"])
        .sort_values("date", kind="mergesort")
        .set_index("date")
        .rename(columns={"pred": f"Pred: {variant}"})
    )
    plot_df = plot_df.join(pred_series, how="left")
    aligned = actual_series.join(pred_series, how="inner").dropna()
    if not aligned.empty:
        error_df[f"Forecast Error: {variant}"] = aligned.iloc[:, 1] - aligned["Actual"]

st.line_chart(plot_df, use_container_width=True)

st.subheader("Forecast Error")
if error_df.empty:
    st.info("Forecast Error series unavailable for selected variants.")
else:
    st.line_chart(error_df, use_container_width=True)

st.subheader("Directional Accuracy")
dir_rows: list[dict[str, object]] = []
for (split_name, model_variant), group in predictions.groupby(["split", "model_variant"], dropna=False):
    dir_rows.append(
        {
            "split": split_name,
            "model_variant": model_variant,
            "directional_accuracy": directional_accuracy(group),
        }
    )
dir_df = pd.DataFrame(dir_rows).sort_values(["split", "model_variant"], kind="mergesort")
st.dataframe(dir_df, hide_index=True, use_container_width=True)

st.info("Experimental forecasting workflow for research purposes. Not investment advice.")
