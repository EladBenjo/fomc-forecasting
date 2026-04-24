"""Forecast and model-results page for economist dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.artifacts import remediation_commands
from app.lib.cached_data import get_phase4_payload, get_phase4_runs
from app.lib.dashboard_metrics import directional_accuracy

st.title("Forecast & Model Results")
st.caption("Model performance, Forecast Error diagnostics, and Baseline vs Text-Enhanced Model comparisons.")

commands = remediation_commands()
runs = get_phase4_runs(require_complete=True)
if not runs:
    st.warning("No complete model run artifacts found under `data/models/baselines/t5yie`.")
    st.code(commands["phase4"])
    st.stop()

selected_run = st.selectbox("Run version", options=runs, index=len(runs) - 1)
try:
    payload = get_phase4_payload(selected_run)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code(commands["phase4"])
    st.stop()

predictions = payload["predictions"]
results = payload["results_table"].copy()
paired = payload["paired_comparison"].copy()
summary = payload["run_summary"]

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
