"""Monthly target/feature alignment helpers for Phase 3 dataset building."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd

from datasets.schema.fields import (
    DATE_COLUMN,
    DOC_ID_COLUMN,
    FEATURE_MONTH_COLUMN,
    FEATURE_MONTH_USED_COLUMN,
    MISSING_PERIOD_REASON_COLUMN,
    MISSING_REASON_NO_DOCS_MONTH,
    MISSING_REASON_PRE_FEATURE_HISTORY,
    MISSING_REASON_RESIDUAL_TRUE_GAP,
    MONTHLY_FEATURE_COLUMNS,
    SOURCE_TYPE_COLUMN,
    TARGET_ABS_MEAN_COLUMN,
    TARGET_COLUMN,
    TARGET_MONTH_COLUMN,
    TARGET_OBS_COLUMN,
    TARGET_REQUIRED_COLUMNS,
)

_BASE_MONTHLY_AGGREGATIONS: dict[str, tuple[str, str]] = {
    "hawkish_score": ("hawkish_score", "mean"),
    "novelty": ("novelty", "mean"),
    "n_hawkish": ("n_hawkish", "sum"),
    "n_dovish": ("n_dovish", "sum"),
    "n_neutral": ("n_neutral", "sum"),
    "n_target_sentences": ("n_target_sentences", "sum"),
    "doc_count": (DOC_ID_COLUMN, "count"),
    "text_length_words_max": ("text_length_words", "max"),
}

_EVENT_WINDOWS: tuple[int, ...] = (7, 14, 30)
_EVENT_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"hawkish_score_max_abs_signed_{days}d" for days in _EVENT_WINDOWS
)
_SUPPORTED_MONTHLY_FEATURE_COLUMNS: tuple[str, ...] = (
    *tuple(_BASE_MONTHLY_AGGREGATIONS.keys()),
    "role_share_chairman",
    *_EVENT_FEATURE_COLUMNS,
)

_FEATURE_DEPENDENCY_COLUMNS: dict[str, tuple[str, ...]] = {
    "hawkish_score": ("hawkish_score",),
    "novelty": ("novelty",),
    "n_hawkish": ("n_hawkish",),
    "n_dovish": ("n_dovish",),
    "n_neutral": ("n_neutral",),
    "n_target_sentences": ("n_target_sentences",),
    "doc_count": (DOC_ID_COLUMN,),
    "text_length_words_max": ("text_length_words",),
    "role_share_chairman": (DOC_ID_COLUMN, SOURCE_TYPE_COLUMN, "role"),
    "hawkish_score_max_abs_signed_7d": (DOC_ID_COLUMN, DATE_COLUMN, "hawkish_score"),
    "hawkish_score_max_abs_signed_14d": (DOC_ID_COLUMN, DATE_COLUMN, "hawkish_score"),
    "hawkish_score_max_abs_signed_30d": (DOC_ID_COLUMN, DATE_COLUMN, "hawkish_score"),
}

_WHITESPACE_RE = re.compile(r"\s+")


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, frame_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _validate_selected_feature_dependencies(
    features_df: pd.DataFrame,
    selected: Sequence[str],
) -> None:
    missing_by_feature: dict[str, list[str]] = {}
    for feature_col in selected:
        required = _FEATURE_DEPENDENCY_COLUMNS.get(feature_col, ())
        missing = [col for col in required if col not in features_df.columns]
        if missing:
            missing_by_feature[feature_col] = missing

    if not missing_by_feature:
        return

    details = "; ".join(
        f"{feature_col}: {missing_cols}"
        for feature_col, missing_cols in missing_by_feature.items()
    )
    raise ValueError(
        "features_df is missing required input columns for selected monthly features: "
        f"{details}"
    )


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


def _aggregate_role_share_chairman(features_monthly_source: pd.DataFrame) -> pd.DataFrame:
    speech_rows = features_monthly_source[
        features_monthly_source[SOURCE_TYPE_COLUMN].astype(str).str.upper().str.strip() == "SPEECH"
    ].copy()
    if speech_rows.empty:
        return pd.DataFrame(columns=[FEATURE_MONTH_COLUMN, "role_share_chairman"])

    speech_rows["role_normalized"] = speech_rows["role"].map(_normalize_role_for_chairman_share)
    monthly = (
        speech_rows.groupby(FEATURE_MONTH_COLUMN)
        .agg(
            speech_doc_count=(DOC_ID_COLUMN, "count"),
            chairman_doc_count=("role_normalized", lambda s: (s == "chairman").sum()),
        )
        .reset_index()
    )
    monthly["role_share_chairman"] = np.where(
        monthly["speech_doc_count"] > 0,
        monthly["chairman_doc_count"] / monthly["speech_doc_count"],
        np.nan,
    )
    return monthly[[FEATURE_MONTH_COLUMN, "role_share_chairman"]]


def _aggregate_hawkish_score_max_abs_signed(
    features_monthly_source: pd.DataFrame,
    *,
    window_days: int,
) -> pd.DataFrame:
    value_col = f"hawkish_score_max_abs_signed_{window_days}d"
    rows: list[dict[str, object]] = []
    event_source = features_monthly_source[
        [FEATURE_MONTH_COLUMN, DATE_COLUMN, DOC_ID_COLUMN, "hawkish_score"]
    ].copy()

    for feature_month, month_df in event_source.groupby(FEATURE_MONTH_COLUMN, sort=True):
        month_end = feature_month.to_timestamp(how="end").normalize()
        window_start = month_end - pd.Timedelta(days=window_days - 1)
        window_df = month_df[
            (month_df[DATE_COLUMN] >= window_start) & (month_df[DATE_COLUMN] <= month_end)
        ].copy()

        if window_df.empty:
            rows.append({FEATURE_MONTH_COLUMN: feature_month, value_col: np.nan})
            continue

        winner = (
            window_df.assign(abs_score=window_df["hawkish_score"].abs())
            .sort_values(
                ["abs_score", DATE_COLUMN, DOC_ID_COLUMN],
                ascending=[False, False, False],
                kind="mergesort",
            )
            .iloc[0]
        )
        rows.append({FEATURE_MONTH_COLUMN: feature_month, value_col: float(winner["hawkish_score"])})

    return pd.DataFrame(rows).sort_values(FEATURE_MONTH_COLUMN, kind="mergesort").reset_index(drop=True)


def normalize_dates(
    df: pd.DataFrame,
    *,
    frame_name: str,
    date_column: str = DATE_COLUMN,
    require_unique: bool = False,
) -> pd.DataFrame:
    """Normalize and validate date column for deterministic processing."""
    if date_column not in df.columns:
        raise ValueError(f"{frame_name} is missing required date column: {date_column!r}")

    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce").dt.normalize()
    out = out.dropna(subset=[date_column]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{frame_name} has no valid rows after date normalization.")

    if not out[date_column].is_monotonic_increasing:
        raise ValueError(f"{frame_name}.{date_column} must be monotonic increasing.")
    if require_unique and (not out[date_column].is_unique):
        raise ValueError(f"{frame_name}.{date_column} must be unique.")

    return out.sort_values(date_column, kind="mergesort").reset_index(drop=True)


def resolve_monthly_feature_columns(columns: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return validated monthly feature column selection in deterministic order."""
    if columns is None:
        return MONTHLY_FEATURE_COLUMNS

    requested = tuple(columns)
    invalid = [col for col in requested if col not in _SUPPORTED_MONTHLY_FEATURE_COLUMNS]
    if invalid:
        raise ValueError(f"Unsupported monthly feature columns requested: {invalid}")
    if len(set(requested)) != len(requested):
        raise ValueError("Monthly feature column list contains duplicates.")
    if not requested:
        raise ValueError("At least one monthly feature column must be selected.")
    return requested


def aggregate_target_monthly(
    target_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Aggregate daily target to monthly target-month rows."""
    _require_columns(target_df, TARGET_REQUIRED_COLUMNS, frame_name="target_df")
    if target_column not in target_df.columns:
        raise ValueError(f"target_df is missing configured target column: {target_column!r}")

    out = (
        target_df.assign(**{TARGET_MONTH_COLUMN: target_df[DATE_COLUMN].dt.to_period("M")})
        .groupby(TARGET_MONTH_COLUMN)
        .agg(
            **{
                target_column: (target_column, "mean"),
                TARGET_ABS_MEAN_COLUMN: (target_column, lambda s: s.abs().mean()),
                TARGET_OBS_COLUMN: (target_column, "size"),
            }
        )
        .reset_index()
        .sort_values(TARGET_MONTH_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    if not out[TARGET_MONTH_COLUMN].is_monotonic_increasing:
        raise ValueError("Monthly target index is not monotonic increasing.")
    return out


def aggregate_features_monthly(
    features_df: pd.DataFrame,
    *,
    monthly_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate document-level features into monthly feature-month rows."""
    selected = resolve_monthly_feature_columns(monthly_feature_columns)
    _validate_selected_feature_dependencies(features_df, selected)
    _require_columns(features_df, [DATE_COLUMN], frame_name="features_df")

    monthly_source = features_df.assign(**{FEATURE_MONTH_COLUMN: features_df[DATE_COLUMN].dt.to_period("M")})
    base_selected = [col for col in selected if col in _BASE_MONTHLY_AGGREGATIONS]
    if base_selected:
        agg_kwargs = {out_col: _BASE_MONTHLY_AGGREGATIONS[out_col] for out_col in base_selected}
        out = (
            monthly_source.groupby(FEATURE_MONTH_COLUMN)
            .agg(**agg_kwargs)
            .reset_index()
            .sort_values(FEATURE_MONTH_COLUMN, kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        out = (
            monthly_source[[FEATURE_MONTH_COLUMN]]
            .drop_duplicates()
            .sort_values(FEATURE_MONTH_COLUMN, kind="mergesort")
            .reset_index(drop=True)
        )

    if "role_share_chairman" in selected:
        role_share = _aggregate_role_share_chairman(monthly_source)
        out = out.merge(role_share, on=FEATURE_MONTH_COLUMN, how="left")

    for event_window in _EVENT_WINDOWS:
        event_col = f"hawkish_score_max_abs_signed_{event_window}d"
        if event_col not in selected:
            continue
        event_monthly = _aggregate_hawkish_score_max_abs_signed(
            monthly_source,
            window_days=event_window,
        )
        out = out.merge(event_monthly, on=FEATURE_MONTH_COLUMN, how="left")

    if not out[FEATURE_MONTH_COLUMN].is_monotonic_increasing:
        raise ValueError("Monthly feature index is not monotonic increasing.")
    return out[[FEATURE_MONTH_COLUMN, *selected]]


def align_monthly_no_lookahead(
    target_monthly: pd.DataFrame,
    features_monthly: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    monthly_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Align target month M to feature month M-1 with explicit leakage checks."""
    selected = resolve_monthly_feature_columns(monthly_feature_columns)
    _require_columns(
        target_monthly,
        [TARGET_MONTH_COLUMN, target_column, TARGET_ABS_MEAN_COLUMN, TARGET_OBS_COLUMN],
        frame_name="target_monthly",
    )
    _require_columns(
        features_monthly,
        [FEATURE_MONTH_COLUMN, *selected],
        frame_name="features_monthly",
    )

    out = target_monthly[
        [TARGET_MONTH_COLUMN, target_column, TARGET_ABS_MEAN_COLUMN, TARGET_OBS_COLUMN]
    ].copy()
    out[FEATURE_MONTH_USED_COLUMN] = out[TARGET_MONTH_COLUMN] - 1
    out = (
        out.merge(
            features_monthly,
            left_on=FEATURE_MONTH_USED_COLUMN,
            right_on=FEATURE_MONTH_COLUMN,
            how="left",
        )
        .drop(columns=[FEATURE_MONTH_COLUMN])
        .sort_values(TARGET_MONTH_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )

    if not (out[FEATURE_MONTH_USED_COLUMN] < out[TARGET_MONTH_COLUMN]).all():
        raise ValueError("No-lookahead check failed: found feature_month_used >= target_month.")
    if len(out) != len(target_monthly):
        raise ValueError(
            "Monthly merge row mismatch: "
            f"aligned={len(out)} target_monthly={len(target_monthly)}"
        )
    return out


def add_missing_period_reason(
    aligned_monthly: pd.DataFrame,
    features_monthly: pd.DataFrame,
    *,
    monthly_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Tag each row with notebook-defined missingness reasons."""
    selected = resolve_monthly_feature_columns(monthly_feature_columns)
    _require_columns(
        aligned_monthly,
        [TARGET_MONTH_COLUMN, FEATURE_MONTH_USED_COLUMN, *selected],
        frame_name="aligned_monthly",
    )
    _require_columns(
        features_monthly,
        [FEATURE_MONTH_COLUMN],
        frame_name="features_monthly",
    )

    if "doc_count" in features_monthly.columns:
        months_with_docs = set(
            features_monthly.loc[features_monthly["doc_count"] > 0, FEATURE_MONTH_COLUMN]
        )
    else:
        months_with_docs = set(features_monthly[FEATURE_MONTH_COLUMN])
    if not months_with_docs:
        raise ValueError("No months with docs found in monthly features.")

    first_feature_month = min(months_with_docs)
    out = aligned_monthly.copy()
    out[MISSING_PERIOD_REASON_COLUMN] = np.select(
        [
            out[FEATURE_MONTH_USED_COLUMN] < first_feature_month,
            ~out[FEATURE_MONTH_USED_COLUMN].isin(months_with_docs),
        ],
        [MISSING_REASON_PRE_FEATURE_HISTORY, MISSING_REASON_NO_DOCS_MONTH],
        default=MISSING_REASON_RESIDUAL_TRUE_GAP,
    )

    for feature_col in selected:
        missing_mask = out[feature_col].isna()
        reason_count = out.loc[missing_mask, MISSING_PERIOD_REASON_COLUMN].value_counts()
        if int(reason_count.sum()) != int(missing_mask.sum()):
            raise ValueError(f"Missing decomposition mismatch for {feature_col}.")

    return out


def finalize_monthly_model_dataset(
    monthly_df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    monthly_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Finalize ordering/types for model dataset parquet."""
    selected = resolve_monthly_feature_columns(monthly_feature_columns)
    required = [
        TARGET_MONTH_COLUMN,
        FEATURE_MONTH_USED_COLUMN,
        target_column,
        TARGET_ABS_MEAN_COLUMN,
        TARGET_OBS_COLUMN,
        *selected,
        MISSING_PERIOD_REASON_COLUMN,
    ]
    _require_columns(monthly_df, required, frame_name="monthly_df")

    out = monthly_df.copy()
    out[DATE_COLUMN] = out[TARGET_MONTH_COLUMN].dt.to_timestamp()
    if not out[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Final dataset date index is not monotonic increasing.")
    if not out[TARGET_MONTH_COLUMN].is_monotonic_increasing:
        raise ValueError("Final target_month index is not monotonic increasing.")
    if not (out[FEATURE_MONTH_USED_COLUMN] < out[TARGET_MONTH_COLUMN]).all():
        raise ValueError("No-lookahead integrity check failed during finalization.")

    out[TARGET_MONTH_COLUMN] = out[TARGET_MONTH_COLUMN].astype(str)
    out[FEATURE_MONTH_USED_COLUMN] = out[FEATURE_MONTH_USED_COLUMN].astype(str)
    return out[
        [
            TARGET_MONTH_COLUMN,
            DATE_COLUMN,
            FEATURE_MONTH_USED_COLUMN,
            target_column,
            TARGET_ABS_MEAN_COLUMN,
            TARGET_OBS_COLUMN,
            *selected,
            MISSING_PERIOD_REASON_COLUMN,
        ]
    ].reset_index(drop=True)


def validate_expected_rows(
    actual_rows: Mapping[str, int],
    expected_rows: Mapping[str, int] | None,
) -> None:
    """Optional strict row-count checks for deterministic outputs."""
    if not expected_rows:
        return

    for key, expected in expected_rows.items():
        if key not in actual_rows:
            raise ValueError(f"Unknown expected row check key: {key!r}")
        actual = actual_rows[key]
        if actual != expected:
            raise ValueError(
                f"Row count check failed for {key}: expected={expected}, actual={actual}"
            )


def validate_feature_source_column(features_df: pd.DataFrame) -> None:
    """Guard against accidental schema drift for source_type."""
    _require_columns(features_df, [SOURCE_TYPE_COLUMN], frame_name="features_df")
