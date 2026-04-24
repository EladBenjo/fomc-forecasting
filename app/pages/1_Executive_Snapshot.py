"""Executive Snapshot page for economist dashboard."""

from __future__ import annotations

import streamlit as st

from app.lib.artifacts import remediation_commands
from app.lib.cached_data import (
    get_artifact_freshness,
    get_features_frame,
    get_latest_phase4_payload,
    get_model_dataset_frame,
    get_optional_metadata_frame,
    get_status_snapshot,
)
from app.lib.dashboard_metrics import recent_events_table, recent_window_snapshot

st.title("Executive Snapshot")
st.caption("Fast overview of current Communication Signal, Policy Tone, and forecast availability.")

commands = remediation_commands()
status = get_status_snapshot()

snapshot_col_1, snapshot_col_2, snapshot_col_3, snapshot_col_4 = st.columns(4)
for col, check in zip((snapshot_col_1, snapshot_col_2, snapshot_col_3, snapshot_col_4), status["checks"]):
    with col:
        st.metric(check["label"], "Ready" if check["ready"] else "Missing")

if status["ready_count"] < status["total_checks"]:
    st.warning("Some required artifacts are missing. The dashboard will render available sections only.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Inflation Expectations and Communication Context")
    try:
        model_df = get_model_dataset_frame()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        st.code(commands["model_dataset"])
    else:
        if model_df.empty:
            st.warning("Model dataset exists but has no rows.")
        else:
            ordered = model_df.sort_values("date", kind="mergesort")
            latest_row = ordered.iloc[-1]
            prev_row = ordered.iloc[-2] if len(ordered) > 1 else None

            latest_target = latest_row.get("t5yie_diff1")
            prev_target = prev_row.get("t5yie_diff1") if prev_row is not None else None
            target_delta = (
                float(latest_target - prev_target)
                if latest_target is not None and prev_target is not None
                else None
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Latest Target Date", str(latest_row["date"].date()))
            with c2:
                if latest_target is not None:
                    st.metric(
                        "Recent Change in Inflation Expectations",
                        f"{float(latest_target):.3f}",
                        None if target_delta is None else f"{target_delta:+.3f}",
                    )
            with c3:
                if "hawkish_score" in ordered.columns:
                    st.metric("Latest Communication Signal", f"{float(ordered['hawkish_score'].iloc[-1]):.3f}")

with right:
    st.subheader("Latest Forecast")
    phase4_payload = get_latest_phase4_payload()
    if phase4_payload is None:
        st.warning("No completed forecasting run found.")
        st.code(commands["phase4"])
    else:
        preds = phase4_payload["predictions"]
        if preds.empty:
            st.warning("Predictions artifact is empty.")
        else:
            latest_pred = preds.sort_values("date", kind="mergesort").iloc[-1]
            st.metric("Latest Forecast Date", str(latest_pred["date"].date()))
            st.metric("Latest Model Prediction", f"{float(latest_pred['pred']):.3f}")
            st.caption(f"Model variant: `{latest_pred['model_variant']}`")
            st.caption(f"Run: `{phase4_payload['run_version']}`")

st.subheader("Policy Tone and Recent Event")
lookback_days = st.slider("Trailing window (days)", min_value=30, max_value=365, value=90, step=15)

try:
    features = get_features_frame(prefer_backfilled=True)
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.code(commands["features"])
    st.stop()

if features.empty:
    st.warning("Feature artifact exists but has no rows.")
    st.stop()

recent_snapshot = recent_window_snapshot(features, lookback_days=lookback_days)
if not recent_snapshot:
    st.warning("Unable to compute recent snapshot from available feature rows.")
else:
    policy_cols = st.columns(4)
    with policy_cols[0]:
        st.metric("Window Start", str(recent_snapshot["window_start"].date()))
    with policy_cols[1]:
        st.metric("Latest Communication Date", str(recent_snapshot["latest_date"].date()))
    with policy_cols[2]:
        st.metric("Policy Tone", str(recent_snapshot.get("policy_tone", "Unavailable")))
    with policy_cols[3]:
        st.metric("Recent Document Count", int(recent_snapshot.get("doc_count", 0)))

    strongest = recent_snapshot.get("strongest_event")
    if strongest:
        st.caption(
            "Strongest Recent Event: "
            f"{strongest.get('date').date()} | "
            f"{strongest.get('source_type')} #{strongest.get('doc_id')} | "
            f"hawkish_score={float(strongest.get('hawkish_score')):.3f}"
        )

st.subheader("Top Recent Highlights / Warnings")
metadata = get_optional_metadata_frame()
events = recent_events_table(
    features,
    metadata_df=metadata,
    lookback_days=lookback_days,
    top_n=5,
)
if events.empty:
    st.info("No recent events available for highlights.")
else:
    for _, row in events.iterrows():
        st.write(
            f"- `{row['date'].date()}` {row['source_type']} `{row['title_or_doc']}` | "
            f"Communication Signal: {float(row['hawkish_score']):.3f} | "
            f"Novelty: {float(row['novelty']):.3f} | {row['explanation']}"
        )

st.subheader("Data Freshness")
st.dataframe(get_artifact_freshness(), hide_index=True, use_container_width=True)
