"""Daily no-lookahead target/feature alignment helpers for Phase 3 dataset building."""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd

from datasets.schema.fields import (
    DATE_COLUMN,
    DOC_ID_COLUMN,
    SOURCE_TYPE_COLUMN,
    TARGET_COLUMN,
)

DEFAULT_DAILY_COMM_WINDOWS: tuple[int, ...] = (7, 14, 30)
DEFAULT_DAILY_AR_LAGS: tuple[int, ...] = (1, 2, 5, 10)
DEFAULT_DAILY_AR_ROLL_WINDOWS: tuple[int, ...] = (5, 10)

_WHITESPACE_RE = re.compile(r"\s+")


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, frame_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _normalize_role_for_chairman_share(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    if text.replace(" ", "") == "chairman":
        return "chairman"
    return text


def _validate_windows(window_days: Sequence[int]) -> tuple[int, ...]:
    windows = tuple(int(days) for days in window_days)
    if not windows:
        raise ValueError("At least one communication window is required.")
    if len(set(windows)) != len(windows):
        raise ValueError("Communication windows contain duplicates.")
    if any(days <= 0 for days in windows):
        raise ValueError("Communication windows must be positive integers.")
    return windows


def resolve_daily_ar_feature_columns(target_column: str = TARGET_COLUMN) -> tuple[str, ...]:
    """Return deterministic daily autoregressive feature column names."""
    return (
        f"{target_column}_lag_1",
        f"{target_column}_lag_2",
        f"{target_column}_lag_5",
        f"{target_column}_lag_10",
        f"{target_column}_roll_mean_5",
        f"{target_column}_roll_mean_10",
        f"{target_column}_roll_std_5",
        f"{target_column}_roll_std_10",
    )


def resolve_daily_comm_feature_columns(
    features_df: pd.DataFrame,
    *,
    window_days: Sequence[int] = DEFAULT_DAILY_COMM_WINDOWS,
) -> tuple[str, ...]:
    """Return deterministic trailing-window communication feature names."""
    windows = _validate_windows(window_days)
    has_target_ratio = "target_sentences_ratio" in features_df.columns
    has_text_length = "text_length_words" in features_df.columns
    has_role_share = {"role", SOURCE_TYPE_COLUMN}.issubset(features_df.columns)

    out: list[str] = []
    for days in windows:
        out.extend(
            [
                f"hawkish_score_mean_{days}d",
                f"hawkish_score_sum_{days}d",
                f"hawkish_score_max_abs_signed_{days}d",
                f"novelty_mean_{days}d",
                f"n_target_sentences_sum_{days}d",
                f"doc_count_{days}d",
            ]
        )
        if has_target_ratio:
            out.append(f"target_sentences_ratio_mean_{days}d")
        if has_text_length:
            out.append(f"text_length_words_max_{days}d")
        if has_role_share:
            out.append(f"role_share_chairman_{days}d")

    return tuple(out)


def build_daily_autoregressive_features(
    target_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Build shifted lag/rolling target features that exclude the current target observation."""
    _require_columns(target_df, [DATE_COLUMN, target_column], frame_name="target_df")
    if not target_df[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("target_df.date must be monotonic increasing for autoregressive feature construction.")
    if not target_df[DATE_COLUMN].is_unique:
        raise ValueError("target_df.date must be unique for autoregressive feature construction.")

    out = target_df[[DATE_COLUMN, target_column]].copy()
    target_series = out[target_column]

    for lag in DEFAULT_DAILY_AR_LAGS:
        out[f"{target_column}_lag_{lag}"] = target_series.shift(lag)

    shifted = target_series.shift(1)
    for window in DEFAULT_DAILY_AR_ROLL_WINDOWS:
        out[f"{target_column}_roll_mean_{window}"] = shifted.rolling(
            window=window,
            min_periods=window,
        ).mean()
        out[f"{target_column}_roll_std_{window}"] = shifted.rolling(
            window=window,
            min_periods=window,
        ).std()

    return out


def _event_hawkish_score_max_abs_signed(window_df: pd.DataFrame) -> float:
    candidates = window_df[[DATE_COLUMN, DOC_ID_COLUMN, "hawkish_score"]].dropna(
        subset=["hawkish_score"]
    )
    if candidates.empty:
        return float("nan")

    winner = (
        candidates.assign(abs_score=candidates["hawkish_score"].abs())
        .sort_values(
            ["abs_score", DATE_COLUMN, DOC_ID_COLUMN],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .iloc[0]
    )
    return float(winner["hawkish_score"])


def _role_share_chairman(window_df: pd.DataFrame) -> float:
    speech_rows = window_df[
        window_df[SOURCE_TYPE_COLUMN].astype(str).str.upper().str.strip() == "SPEECH"
    ].copy()
    if speech_rows.empty:
        return float("nan")

    speech_rows["role_normalized"] = speech_rows["role"].map(_normalize_role_for_chairman_share)
    speech_doc_count = len(speech_rows)
    chairman_doc_count = int((speech_rows["role_normalized"] == "chairman").sum())
    return float(chairman_doc_count) / float(speech_doc_count)


def aggregate_communication_features_trailing_daily(
    target_dates: pd.Series | pd.DatetimeIndex,
    features_df: pd.DataFrame,
    *,
    window_days: Sequence[int] = DEFAULT_DAILY_COMM_WINDOWS,
    lag_days: int = 0,
) -> pd.DataFrame:
    """
    Build trailing-window communication features for each target date.

    For each target date ``t`` and each window ``w`` in days, window rows satisfy:
    ``doc_date >= t - lag_days - w + 1`` and ``doc_date <= t - lag_days``.
    """
    if lag_days < 0:
        raise ValueError("lag_days must be >= 0.")
    windows = _validate_windows(window_days)

    required = [
        DATE_COLUMN,
        DOC_ID_COLUMN,
        "hawkish_score",
        "novelty",
        "n_target_sentences",
    ]
    _require_columns(features_df, required, frame_name="features_df")

    has_target_ratio = "target_sentences_ratio" in features_df.columns
    has_text_length = "text_length_words" in features_df.columns
    has_role_share = {"role", SOURCE_TYPE_COLUMN}.issubset(features_df.columns)

    features_source = features_df.copy()
    features_source = features_source.sort_values(
        [DATE_COLUMN, DOC_ID_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)

    normalized_target_dates = pd.to_datetime(target_dates, errors="coerce")
    if isinstance(normalized_target_dates, pd.Series):
        normalized_target_dates = normalized_target_dates.dt.normalize()
    else:
        normalized_target_dates = normalized_target_dates.normalize()
    if pd.isna(normalized_target_dates).any():
        raise ValueError("target_dates contains invalid values.")

    rows: list[dict[str, object]] = []
    for target_date in normalized_target_dates:
        row: dict[str, object] = {DATE_COLUMN: target_date}
        window_end = target_date - pd.Timedelta(days=lag_days)

        for days in windows:
            window_start = window_end - pd.Timedelta(days=days - 1)
            window_df = features_source[
                (features_source[DATE_COLUMN] >= window_start)
                & (features_source[DATE_COLUMN] <= window_end)
            ].copy()

            if not window_df.empty:
                if window_df[DATE_COLUMN].max() > window_end:
                    raise ValueError("No-lookahead failed: found doc_date > window end.")
                if window_df[DATE_COLUMN].min() < window_start:
                    raise ValueError("Window construction failed: found doc_date < window start.")

            row[f"hawkish_score_mean_{days}d"] = float(window_df["hawkish_score"].mean())
            row[f"hawkish_score_sum_{days}d"] = float(window_df["hawkish_score"].fillna(0).sum())
            row[f"hawkish_score_max_abs_signed_{days}d"] = _event_hawkish_score_max_abs_signed(
                window_df
            )
            row[f"novelty_mean_{days}d"] = float(window_df["novelty"].mean())
            row[f"n_target_sentences_sum_{days}d"] = float(window_df["n_target_sentences"].fillna(0).sum())
            row[f"doc_count_{days}d"] = int(len(window_df))

            if has_target_ratio:
                row[f"target_sentences_ratio_mean_{days}d"] = float(
                    window_df["target_sentences_ratio"].mean()
                )
            if has_text_length:
                row[f"text_length_words_max_{days}d"] = float(window_df["text_length_words"].max())
            if has_role_share:
                row[f"role_share_chairman_{days}d"] = _role_share_chairman(window_df)

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(DATE_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )


def validate_daily_autoregressive_no_lookahead(
    dataset_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Validate that autoregressive lag/rolling features are strictly shift-first."""
    _require_columns(dataset_df, [DATE_COLUMN, target_column], frame_name="dataset_df")
    for col in resolve_daily_ar_feature_columns(target_column):
        if col not in dataset_df.columns:
            raise ValueError(f"dataset_df is missing required autoregressive feature column: {col!r}")

    lag_1_col = f"{target_column}_lag_1"
    expected_lag_1 = dataset_df[target_column].shift(1)
    if not dataset_df[lag_1_col].equals(expected_lag_1):
        raise ValueError(f"No-lookahead failed for {lag_1_col}.")

    for lag in DEFAULT_DAILY_AR_LAGS:
        col = f"{target_column}_lag_{lag}"
        expected = dataset_df[target_column].shift(lag)
        if not dataset_df[col].equals(expected):
            raise ValueError(f"No-lookahead failed for {col}.")

    shifted = dataset_df[target_column].shift(1)
    for window in DEFAULT_DAILY_AR_ROLL_WINDOWS:
        mean_col = f"{target_column}_roll_mean_{window}"
        std_col = f"{target_column}_roll_std_{window}"
        expected_mean = shifted.rolling(window=window, min_periods=window).mean()
        expected_std = shifted.rolling(window=window, min_periods=window).std()
        if not dataset_df[mean_col].equals(expected_mean):
            raise ValueError(f"No-lookahead failed for {mean_col}.")
        if not dataset_df[std_col].equals(expected_std):
            raise ValueError(f"No-lookahead failed for {std_col}.")


def build_daily_model_dataset_frame(
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    communication_windows: Sequence[int] = DEFAULT_DAILY_COMM_WINDOWS,
    communication_lag_days: int = 0,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build daily model dataset frame with AR and trailing communication features."""
    windows = _validate_windows(communication_windows)
    target_ar = build_daily_autoregressive_features(target_df, target_column=target_column)
    communication = aggregate_communication_features_trailing_daily(
        target_ar[DATE_COLUMN],
        features_df,
        window_days=windows,
        lag_days=communication_lag_days,
    )

    out = (
        target_ar.merge(communication, on=DATE_COLUMN, how="left")
        .sort_values(DATE_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(out) != len(target_df):
        raise ValueError(
            "Daily merge row mismatch: "
            f"aligned={len(out)} target_df={len(target_df)}"
        )
    if not out[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Daily dataset date index is not monotonic increasing.")
    if not out[DATE_COLUMN].is_unique:
        raise ValueError("Daily dataset date index is not unique.")

    validate_daily_autoregressive_no_lookahead(out, target_column=target_column)

    feature_columns = tuple(col for col in out.columns if col not in {DATE_COLUMN, target_column})
    return out, feature_columns


__all__ = [
    "DEFAULT_DAILY_COMM_WINDOWS",
    "build_daily_autoregressive_features",
    "aggregate_communication_features_trailing_daily",
    "resolve_daily_ar_feature_columns",
    "resolve_daily_comm_feature_columns",
    "validate_daily_autoregressive_no_lookahead",
    "build_daily_model_dataset_frame",
]
