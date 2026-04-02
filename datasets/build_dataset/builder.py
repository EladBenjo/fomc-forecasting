"""Production builder for the monthly leak-free t5yie_diff1 modeling dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Phase 3 monthly leak-free modeling dataset for t5yie_diff1."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to doc-level feature parquet (default: data/features/doc_level/features.parquet).",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Path to target parquet (default: data/targets/t5yie_diff1.parquet).",
    )
    parser.add_argument(
        "--output-dataset-path",
        type=str,
        default=None,
        help="Path to output model dataset parquet (default: data/targets/model_dataset_t5yie.parquet).",
    )
    parser.add_argument(
        "--split-output-path",
        type=str,
        default=None,
        help="Path to output time-split json (default: data/splits/time_splits.json).",
    )
    parser.add_argument(
        "--summary-output-path",
        type=str,
        default=None,
        help="Optional path to write summary JSON.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=TARGET_COLUMN,
        help=f"Target column name (default: {TARGET_COLUMN}).",
    )
    parser.add_argument(
        "--monthly-feature-columns",
        type=str,
        default=",".join(MONTHLY_FEATURE_COLUMNS),
        help="Comma-separated monthly feature columns in output order.",
    )
    parser.add_argument(
        "--expected-rows-json",
        type=str,
        default=None,
        help='Optional JSON object for strict row checks (e.g. \'{"dataset_rows": 279}\').',
    )
    parser.add_argument("--train-end", type=str, default="2016-12-31", help="Train split end date (inclusive).")
    parser.add_argument("--val-start", type=str, default="2017-01-01", help="Validation split start date (inclusive).")
    parser.add_argument("--val-end", type=str, default="2020-12-31", help="Validation split end date (inclusive).")
    parser.add_argument("--test-start", type=str, default="2021-01-01", help="Test split start date (inclusive).")
    return parser


def _parse_expected_rows_json(raw_json: str | None) -> dict[str, int] | None:
    if raw_json is None:
        return None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --expected-rows-json payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--expected-rows-json must decode to a JSON object.")

    out: dict[str, int] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError("--expected-rows-json keys must be strings.")
        try:
            out[key] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--expected-rows-json value for key {key!r} must be int-like.") from exc
    return out


def _config_from_args(args: argparse.Namespace) -> BuildDatasetConfig:
    monthly_feature_columns = tuple(
        part.strip() for part in args.monthly_feature_columns.split(",") if part.strip()
    )
    if not monthly_feature_columns:
        raise ValueError("At least one --monthly-feature-columns value is required.")

    config = BuildDatasetConfig(
        target_column=args.target_column,
        monthly_feature_columns=monthly_feature_columns,
        expected_rows=_parse_expected_rows_json(args.expected_rows_json),
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    )
    if args.features_path:
        config.features_path = Path(args.features_path)
    if args.target_path:
        config.target_path = Path(args.target_path)
    if args.output_dataset_path:
        config.output_dataset_path = Path(args.output_dataset_path)
    if args.split_output_path:
        config.split_output_path = Path(args.split_output_path)
    if args.summary_output_path:
        config.summary_output_path = Path(args.summary_output_path)

    return config


def main(argv: list[str] | None = None) -> Path:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    output_path = build_model_dataset(config)
    print(f"wrote model dataset: {output_path}")
    print(f"wrote split artifact: {config.split_output_path}")
    if config.summary_output_path:
        print(f"wrote summary artifact: {config.summary_output_path}")
    return output_path


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = ["BuildDatasetConfig", "build_model_dataset", "main"]
