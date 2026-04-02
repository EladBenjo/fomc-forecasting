"""Production builder for the monthly leak-free t5yie_diff1 modeling dataset."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasets.build_dataset.alignment import (
    add_missing_period_reason,
    aggregate_features_monthly,
    aggregate_target_monthly,
    align_monthly_no_lookahead,
    finalize_monthly_model_dataset,
    normalize_dates,
    validate_expected_rows,
    validate_feature_source_column,
)
from datasets.schema.fields import DATE_COLUMN, MONTHLY_FEATURE_COLUMNS, TARGET_COLUMN


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class BuildDatasetConfig:
    """Config for deterministic model dataset + split artifact generation."""

    features_path: Path = field(
        default_factory=lambda: _repo_root() / "data" / "features" / "doc_level" / "features.parquet"
    )
    target_path: Path = field(
        default_factory=lambda: _repo_root() / "data" / "targets" / "t5yie_diff1.parquet"
    )
    output_dataset_path: Path = field(
        default_factory=lambda: _repo_root() / "data" / "targets" / "model_dataset_t5yie.parquet"
    )
    split_output_path: Path = field(
        default_factory=lambda: _repo_root() / "data" / "splits" / "time_splits.json"
    )
    summary_output_path: Path | None = None

    target_column: str = TARGET_COLUMN
    monthly_feature_columns: tuple[str, ...] = MONTHLY_FEATURE_COLUMNS
    expected_rows: dict[str, int] | None = None

    train_end: str = "2016-12-31"
    val_start: str = "2017-01-01"
    val_end: str = "2020-12-31"
    test_start: str = "2021-01-01"


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def _atomic_write_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, out_path)


def _parse_split_boundaries(config: BuildDatasetConfig) -> dict[str, pd.Timestamp]:
    boundaries = {
        "train_end": pd.Timestamp(config.train_end).normalize(),
        "val_start": pd.Timestamp(config.val_start).normalize(),
        "val_end": pd.Timestamp(config.val_end).normalize(),
        "test_start": pd.Timestamp(config.test_start).normalize(),
    }
    if not (boundaries["train_end"] < boundaries["val_start"]):
        raise ValueError("Invalid split config: train_end must be before val_start.")
    if not (boundaries["val_start"] <= boundaries["val_end"]):
        raise ValueError("Invalid split config: val_start must be <= val_end.")
    if not (boundaries["val_end"] < boundaries["test_start"]):
        raise ValueError("Invalid split config: val_end must be before test_start.")
    return boundaries


def _build_split_masks(
    dataset_df: pd.DataFrame,
    boundaries: dict[str, pd.Timestamp],
) -> dict[str, pd.Series]:
    if DATE_COLUMN not in dataset_df.columns:
        raise ValueError(f"Model dataset is missing required column: {DATE_COLUMN!r}")

    date_series = pd.to_datetime(dataset_df[DATE_COLUMN], errors="coerce").dt.normalize()
    if date_series.isna().any():
        raise ValueError("Model dataset contains invalid split dates.")

    masks = {
        "train": date_series <= boundaries["train_end"],
        "val": (date_series >= boundaries["val_start"]) & (date_series <= boundaries["val_end"]),
        "test": date_series >= boundaries["test_start"],
    }

    overlap = (masks["train"] & masks["val"]) | (masks["train"] & masks["test"]) | (masks["val"] & masks["test"])
    if bool(overlap.any()):
        raise ValueError("Split masks overlap; split boundaries are not mutually exclusive.")

    covered = masks["train"] | masks["val"] | masks["test"]
    if not bool(covered.all()):
        uncovered = int((~covered).sum())
        raise ValueError(f"Split masks do not cover all rows. Uncovered rows: {uncovered}")

    return masks


def _build_split_payload(
    dataset_df: pd.DataFrame,
    *,
    boundaries: dict[str, pd.Timestamp],
    masks: dict[str, pd.Series],
) -> tuple[dict[str, Any], dict[str, int]]:
    date_as_str = pd.to_datetime(dataset_df[DATE_COLUMN]).dt.strftime("%Y-%m-%d")

    split_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    split_dates = {name: date_as_str.loc[mask].tolist() for name, mask in masks.items()}

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "date_column": DATE_COLUMN,
        "target_column": TARGET_COLUMN,
        "boundaries": {
            "train_end": boundaries["train_end"].strftime("%Y-%m-%d"),
            "val_start": boundaries["val_start"].strftime("%Y-%m-%d"),
            "val_end": boundaries["val_end"].strftime("%Y-%m-%d"),
            "test_start": boundaries["test_start"].strftime("%Y-%m-%d"),
        },
        "counts": split_counts,
        "dates": split_dates,
    }
    return payload, split_counts


def _build_summary(
    dataset_df: pd.DataFrame,
    *,
    split_counts: dict[str, int],
    monthly_feature_columns: tuple[str, ...],
) -> dict[str, Any]:
    missingness = {
        col: float(dataset_df[col].isna().mean())
        for col in monthly_feature_columns
        if col in dataset_df.columns
    }
    return {
        "rows": int(len(dataset_df)),
        "date_min": pd.to_datetime(dataset_df[DATE_COLUMN]).min().strftime("%Y-%m-%d"),
        "date_max": pd.to_datetime(dataset_df[DATE_COLUMN]).max().strftime("%Y-%m-%d"),
        "split_counts": split_counts,
        "missingness_rate": missingness,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"model_dataset rows: {summary['rows']}")
    print(f"model_dataset date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"split counts: {summary['split_counts']}")
    print(f"missingness_rate: {summary['missingness_rate']}")


def build_model_dataset(config: BuildDatasetConfig) -> Path:
    """Build monthly M-1->M t5yie_diff1 modeling dataset and split artifact."""
    features_path = _as_path(config.features_path)
    target_path = _as_path(config.target_path)
    output_dataset_path = _as_path(config.output_dataset_path)
    split_output_path = _as_path(config.split_output_path)
    summary_output_path = _as_path(config.summary_output_path) if config.summary_output_path else None

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet: {features_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target parquet: {target_path}")

    raw_target_df = pd.read_parquet(target_path)
    raw_features_df = pd.read_parquet(features_path)

    target_df = normalize_dates(raw_target_df, frame_name="target_df", require_unique=True)
    features_df = normalize_dates(raw_features_df, frame_name="features_df", require_unique=False)
    validate_feature_source_column(features_df)

    target_monthly = aggregate_target_monthly(target_df, target_column=config.target_column)
    features_monthly = aggregate_features_monthly(
        features_df,
        monthly_feature_columns=config.monthly_feature_columns,
    )
    aligned = align_monthly_no_lookahead(
        target_monthly,
        features_monthly,
        target_column=config.target_column,
        monthly_feature_columns=config.monthly_feature_columns,
    )
    aligned = add_missing_period_reason(
        aligned,
        features_monthly,
        monthly_feature_columns=config.monthly_feature_columns,
    )
    model_dataset_df = finalize_monthly_model_dataset(
        aligned,
        target_column=config.target_column,
        monthly_feature_columns=config.monthly_feature_columns,
    )

    actual_rows = {
        "target_rows": len(target_df),
        "features_rows": len(features_df),
        "target_monthly_rows": len(target_monthly),
        "features_monthly_rows": len(features_monthly),
        "dataset_rows": len(model_dataset_df),
    }
    validate_expected_rows(actual_rows, config.expected_rows)

    boundaries = _parse_split_boundaries(config)
    masks = _build_split_masks(model_dataset_df, boundaries)
    split_payload, split_counts = _build_split_payload(
        model_dataset_df,
        boundaries=boundaries,
        masks=masks,
    )
    summary = _build_summary(
        model_dataset_df,
        split_counts=split_counts,
        monthly_feature_columns=config.monthly_feature_columns,
    )

    _atomic_write_parquet(model_dataset_df, output_dataset_path)
    _atomic_write_json(split_payload, split_output_path)
    if summary_output_path:
        _atomic_write_json(summary, summary_output_path)

    _print_summary(summary)
    return output_dataset_path


__all__ = ["BuildDatasetConfig", "build_model_dataset"]

