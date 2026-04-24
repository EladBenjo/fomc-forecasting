"""Shared metric/aggregation helpers for Streamlit economist dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_policy_tone(score: float | None) -> str:
    """Map average hawkish score to a business-facing policy tone label."""
    if score is None or pd.isna(score):
        return "Unavailable"
    if score >= 0.15:
        return "Hawkish"
    if score <= -0.15:
        return "Dovish"
    return "Balanced"


def monthly_communication_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate document-level communications into monthly analyst-facing metrics."""
    if features_df.empty or "date" not in features_df.columns:
        return pd.DataFrame()

    work = features_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    if work.empty:
        return pd.DataFrame()

    work["feature_month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    if "doc_id" not in work.columns:
        work["doc_id"] = np.arange(len(work))

    agg_spec: dict[str, tuple[str, str]] = {
        "doc_count": ("doc_id", "count"),
    }
    if "hawkish_score" in work.columns:
        agg_spec["hawkish_score_mean"] = ("hawkish_score", "mean")
    if "novelty" in work.columns:
        agg_spec["novelty_mean"] = ("novelty", "mean")
    if "target_sentences_ratio" in work.columns:
        agg_spec["target_sentences_ratio_mean"] = ("target_sentences_ratio", "mean")
    if "text_length_words" in work.columns:
        agg_spec["text_length_words_mean"] = ("text_length_words", "mean")
        agg_spec["text_length_words_median"] = ("text_length_words", "median")
        agg_spec["text_length_words_max"] = ("text_length_words", "max")

    monthly = work.groupby("feature_month", as_index=False).agg(**agg_spec)

    if "source_type" in work.columns:
        speech_counts = (
            work.assign(is_speech=work["source_type"].astype(str).str.lower().eq("speech"))
            .groupby("feature_month", as_index=False)["is_speech"]
            .sum()
            .rename(columns={"is_speech": "speech_doc_count"})
        )
        monthly = monthly.merge(speech_counts, on="feature_month", how="left")

    if "hawkish_score" in work.columns:
        ranked = work.assign(hawkish_abs=work["hawkish_score"].abs().fillna(0.0)).sort_values(
            ["feature_month", "hawkish_abs", "date", "doc_id"],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
        strongest = (
            ranked.drop_duplicates(subset=["feature_month"], keep="first")
            .loc[:, ["feature_month", "hawkish_score"]]
            .rename(columns={"hawkish_score": "hawkish_score_max_abs_signed"})
        )
        monthly = monthly.merge(strongest, on="feature_month", how="left")

    return monthly.sort_values("feature_month", kind="mergesort").reset_index(drop=True)


def recent_window_snapshot(features_df: pd.DataFrame, *, lookback_days: int = 90) -> dict[str, object]:
    """Summarize recent communication signals over a trailing window."""
    if features_df.empty or "date" not in features_df.columns:
        return {}

    work = features_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    if work.empty:
        return {}

    latest_date = work["date"].max()
    window_start = latest_date - pd.Timedelta(days=max(lookback_days - 1, 0))
    recent = work[(work["date"] >= window_start) & (work["date"] <= latest_date)].copy()
    if recent.empty:
        return {}

    snapshot: dict[str, object] = {
        "latest_date": latest_date,
        "window_start": window_start,
        "doc_count": int(len(recent)),
    }
    if "hawkish_score" in recent.columns:
        avg_hawkish = float(recent["hawkish_score"].mean())
        snapshot["hawkish_score_mean"] = avg_hawkish
        snapshot["policy_tone"] = classify_policy_tone(avg_hawkish)
    if "novelty" in recent.columns:
        snapshot["novelty_mean"] = float(recent["novelty"].mean())
    if "target_sentences_ratio" in recent.columns:
        snapshot["target_sentences_ratio_mean"] = float(recent["target_sentences_ratio"].mean())

    if "hawkish_score" in recent.columns:
        if "doc_id" not in recent.columns:
            recent["doc_id"] = np.arange(len(recent))
        strongest = recent.assign(hawkish_abs=recent["hawkish_score"].abs().fillna(0.0)).sort_values(
            ["hawkish_abs", "date", "doc_id"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        top = strongest.iloc[0]
        snapshot["strongest_event"] = {
            "date": top.get("date"),
            "source_type": top.get("source_type"),
            "doc_id": top.get("doc_id"),
            "hawkish_score": top.get("hawkish_score"),
            "novelty": top.get("novelty"),
        }

    return snapshot


def _event_explanation(
    row: pd.Series,
    *,
    novelty_spike_threshold: float,
    hawkish_extreme_threshold: float,
) -> str:
    novelty_val = row.get("novelty")
    hawkish_abs = abs(float(row.get("hawkish_score", 0.0)))
    novelty_spike = pd.notna(novelty_val) and float(novelty_val) >= novelty_spike_threshold
    tone_extreme = hawkish_abs >= hawkish_extreme_threshold
    if tone_extreme and novelty_spike:
        return "Extreme policy tone with elevated novelty."
    if tone_extreme:
        return "Extreme policy tone reading."
    if novelty_spike:
        return "Novelty Spike in communication."
    return "Recent communication signal."


def recent_events_table(
    features_df: pd.DataFrame,
    *,
    metadata_df: pd.DataFrame | None = None,
    lookback_days: int = 180,
    top_n: int = 25,
) -> pd.DataFrame:
    """Build recent-event table with optional metadata enrichment."""
    if features_df.empty or "date" not in features_df.columns:
        return pd.DataFrame()

    work = features_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    if work.empty:
        return pd.DataFrame()

    if lookback_days > 0:
        latest_date = work["date"].max()
        min_date = latest_date - pd.Timedelta(days=max(lookback_days - 1, 0))
        work = work[work["date"] >= min_date].copy()
    if work.empty:
        return pd.DataFrame()

    if "doc_id" not in work.columns:
        work["doc_id"] = np.arange(len(work))
    if "source_type" not in work.columns:
        work["source_type"] = "unknown"
    if "hawkish_score" not in work.columns:
        work["hawkish_score"] = np.nan
    if "novelty" not in work.columns:
        work["novelty"] = np.nan

    work["source_type"] = work["source_type"].astype(str).str.lower()
    work["doc_id"] = pd.to_numeric(work["doc_id"], errors="coerce").astype("Int64")
    work["hawkish_abs"] = work["hawkish_score"].abs().fillna(0.0)

    if metadata_df is not None and not metadata_df.empty:
        meta = metadata_df.copy()
        meta["source_type"] = meta["source_type"].astype(str).str.lower()
        meta["doc_id"] = pd.to_numeric(meta["doc_id"], errors="coerce").astype("Int64")
        work = work.merge(
            meta[["source_type", "doc_id", "title", "speaker", "event"]],
            on=["source_type", "doc_id"],
            how="left",
        )

    novelty_spike_threshold = float(work["novelty"].quantile(0.90)) if work["novelty"].notna().any() else np.inf
    hawkish_extreme_threshold = float(work["hawkish_abs"].quantile(0.90))
    work["explanation"] = work.apply(
        lambda row: _event_explanation(
            row,
            novelty_spike_threshold=novelty_spike_threshold,
            hawkish_extreme_threshold=hawkish_extreme_threshold,
        ),
        axis=1,
    )

    out = work.sort_values(
        ["date", "hawkish_abs", "doc_id"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    out["title"] = out.get("title", pd.Series(index=out.index)).fillna("")
    out["title_or_doc"] = out["title"].where(out["title"].str.strip().ne(""), out["doc_id"].astype(str))

    return out[
        [
            "date",
            "source_type",
            "doc_id",
            "title_or_doc",
            "hawkish_score",
            "novelty",
            "explanation",
        ]
    ].head(top_n).reset_index(drop=True)


def detect_regime_markers(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Detect transparent heuristic markers for communication regime changes."""
    if monthly_df.empty or "feature_month" not in monthly_df.columns:
        return pd.DataFrame(columns=["feature_month", "marker_type", "metric_value", "explanation"])

    work = monthly_df.copy().sort_values("feature_month", kind="mergesort").reset_index(drop=True)
    markers: list[dict[str, object]] = []

    if "novelty_mean" in work.columns:
        novelty_roll_mean = work["novelty_mean"].rolling(window=12, min_periods=6).mean()
        novelty_roll_std = work["novelty_mean"].rolling(window=12, min_periods=6).std()
        novelty_spike = work["novelty_mean"] > (novelty_roll_mean + 1.5 * novelty_roll_std)
        for _, row in work.loc[novelty_spike.fillna(False)].iterrows():
            markers.append(
                {
                    "feature_month": row["feature_month"],
                    "marker_type": "Novelty Spike",
                    "metric_value": row["novelty_mean"],
                    "explanation": "Novelty exceeds rolling baseline by >1.5 standard deviations.",
                }
            )

    if "hawkish_score_mean" in work.columns:
        hawkish_diff_abs = work["hawkish_score_mean"].diff().abs()
        diff_roll_std = hawkish_diff_abs.rolling(window=12, min_periods=6).std()
        tone_shift = hawkish_diff_abs > (1.5 * diff_roll_std)
        for _, row in work.loc[tone_shift.fillna(False)].iterrows():
            markers.append(
                {
                    "feature_month": row["feature_month"],
                    "marker_type": "Policy Tone Shift",
                    "metric_value": row["hawkish_score_mean"],
                    "explanation": "Month-over-month policy tone change exceeds rolling volatility.",
                }
            )

    if not markers:
        return pd.DataFrame(columns=["feature_month", "marker_type", "metric_value", "explanation"])

    return pd.DataFrame(markers).sort_values(
        ["feature_month", "marker_type"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def directional_accuracy(predictions_df: pd.DataFrame) -> float:
    """Compute directional accuracy (sign agreement), excluding zero actual changes."""
    required = {"actual", "pred"}
    if predictions_df.empty or not required.issubset(predictions_df.columns):
        return float("nan")
    valid = predictions_df[["actual", "pred"]].dropna()
    if valid.empty:
        return float("nan")
    actual_sign = np.sign(valid["actual"].to_numpy(dtype=float))
    pred_sign = np.sign(valid["pred"].to_numpy(dtype=float))
    mask = actual_sign != 0
    if not mask.any():
        return float("nan")
    return float((actual_sign[mask] == pred_sign[mask]).mean())


def feature_correlation_table(
    model_df: pd.DataFrame,
    *,
    target_col: str = "t5yie_diff1",
    candidate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Pearson/Spearman correlation diagnostics against target."""
    if model_df.empty or target_col not in model_df.columns:
        return pd.DataFrame()

    if candidate_cols is None:
        numeric_cols = model_df.select_dtypes(include=["number"]).columns.tolist()
        candidate_cols = [c for c in numeric_cols if c != target_col]

    rows: list[dict[str, object]] = []
    for col in candidate_cols:
        if col not in model_df.columns:
            continue
        sample = model_df[[target_col, col]].dropna()
        if len(sample) < 3:
            continue
        pearson = sample[target_col].corr(sample[col], method="pearson")
        spearman = sample[target_col].corr(sample[col], method="spearman")
        rows.append(
            {
                "feature": col,
                "n_obs": int(len(sample)),
                "pearson": float(pearson),
                "spearman": float(spearman),
                "abs_spearman": abs(float(spearman)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False, kind="mergesort").reset_index(drop=True)


def recent_feature_zscores(
    model_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    periods: int = 24,
) -> pd.DataFrame:
    """Compute recent z-score trajectories for selected monthly features."""
    if model_df.empty or "date" not in model_df.columns:
        return pd.DataFrame()

    available = [col for col in feature_cols if col in model_df.columns]
    if not available:
        return pd.DataFrame()

    work = model_df[["date"] + available].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"]).sort_values("date", kind="mergesort").tail(periods).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    out = pd.DataFrame({"date": work["date"]})
    for col in available:
        series = pd.to_numeric(work[col], errors="coerce")
        mean = series.mean()
        std = series.std()
        if pd.isna(std) or std == 0:
            out[col] = np.nan
        else:
            out[col] = (series - mean) / std
    return out
