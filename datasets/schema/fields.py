"""Canonical schema contracts for the Phase 3 modeling dataset."""

from __future__ import annotations

from typing import Final

DATE_COLUMN: Final[str] = "date"
DOC_ID_COLUMN: Final[str] = "doc_id"
SOURCE_TYPE_COLUMN: Final[str] = "source_type"

TARGET_COLUMN: Final[str] = "t5yie_diff1"
TARGET_MONTH_COLUMN: Final[str] = "target_month"
FEATURE_MONTH_COLUMN: Final[str] = "feature_month"
FEATURE_MONTH_USED_COLUMN: Final[str] = "feature_month_used"

TARGET_ABS_MEAN_COLUMN: Final[str] = "t5yie_diff1_abs_mean"
TARGET_OBS_COLUMN: Final[str] = "t5yie_obs"

MISSING_PERIOD_REASON_COLUMN: Final[str] = "missing_period_reason"
MISSING_REASON_PRE_FEATURE_HISTORY: Final[str] = "pre_feature_history"
MISSING_REASON_NO_DOCS_MONTH: Final[str] = "no_docs_month"
MISSING_REASON_RESIDUAL_TRUE_GAP: Final[str] = "residual_true_gap"

FEATURE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    DOC_ID_COLUMN,
    SOURCE_TYPE_COLUMN,
    DATE_COLUMN,
    "hawkish_score",
    "n_hawkish",
    "n_dovish",
    "n_neutral",
    "n_target_sentences",
    "novelty",
)

TARGET_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    DATE_COLUMN,
    TARGET_COLUMN,
)

MONTHLY_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "hawkish_score",
    "novelty",
    "n_hawkish",
    "n_dovish",
    "n_neutral",
    "n_target_sentences",
    "doc_count",
    "text_length_words_max",
    "role_share_chairman",
    "hawkish_score_max_abs_signed_7d",
    "hawkish_score_max_abs_signed_14d",
    "hawkish_score_max_abs_signed_30d",
)

MODEL_DATASET_COLUMNS: Final[tuple[str, ...]] = (
    TARGET_MONTH_COLUMN,
    DATE_COLUMN,
    FEATURE_MONTH_USED_COLUMN,
    TARGET_COLUMN,
    TARGET_ABS_MEAN_COLUMN,
    TARGET_OBS_COLUMN,
    *MONTHLY_FEATURE_COLUMNS,
    MISSING_PERIOD_REASON_COLUMN,
)
