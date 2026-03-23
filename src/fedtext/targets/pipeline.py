"""
Target-series pipeline entry point.

Fetches a FRED series, materializes canonical raw + transformed parquets under
data/targets, and writes per-run manifest/registry metadata.

Usage:
    python -m fedtext.targets.pipeline
    python -m fedtext.targets.pipeline --series-id T5YIE --transform diff1
    python -m fedtext.targets.pipeline --start 2010-01-01 --end 2020-12-31
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from fedtext.common.paths import REPO_ROOT, TARGETS_DIR
from fedtext.targets.versioning import (
    TargetDatasetManifest,
    default_dataset_version,
    get_git_sha,
    hash_file,
    utc_now_iso,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_SUPPORTED_TRANSFORMS = ("diff1",)
_MANIFEST_REGISTRY = "dataset_registry.sqlite3"


def _series_slug(series_id: str) -> str:
    return series_id.strip().lower()


def _ensure_api_key() -> str:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("Missing FRED_API_KEY in environment.")
    return api_key


def fetch_series(series_id: str, start: str | None, end: str | None) -> pd.DataFrame:
    """
    Fetch a FRED time series as a normalized DataFrame.

    Returns columns:
        - date (datetime64[ns])
        - value (float)
    """
    if not series_id or not series_id.strip():
        raise ValueError("series_id must be a non-empty string.")

    try:
        from fredapi import Fred
    except ImportError as exc:
        raise RuntimeError("fredapi is required to fetch FRED series.") from exc

    fred = Fred(api_key=_ensure_api_key())

    kwargs: dict[str, str] = {}
    if start:
        kwargs["observation_start"] = start
    if end:
        kwargs["observation_end"] = end

    series = fred.get_series(series_id.strip().upper(), **kwargs)
    if series is None or len(series) == 0:
        raise RuntimeError(f"FRED returned no data for series_id={series_id!r}")

    frame = series.rename("value").to_frame().reset_index()
    date_col = frame.columns[0]
    frame = frame.rename(columns={date_col: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return frame[["date", "value"]]


def apply_transform(df: pd.DataFrame, transform_id: str = "diff1") -> pd.DataFrame:
    """
    Apply a target-series transformation.

    Returns columns:
        - date
        - value_transformed
    """
    transform_id = transform_id.strip().lower()
    if transform_id not in _SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unsupported transform_id={transform_id!r}. "
            f"Supported: {', '.join(_SUPPORTED_TRANSFORMS)}"
        )
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Input DataFrame must include columns: date, value.")

    ordered = df[["date", "value"]].copy().sort_values("date")

    if transform_id == "diff1":
        transformed = ordered.assign(value_transformed=ordered["value"].diff())
        transformed = transformed.dropna(subset=["value_transformed"])
        return transformed[["date", "value_transformed"]].reset_index(drop=True)

    raise AssertionError("Unreachable transform branch.")


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def run(
    series_id: str = "T5YIE",
    transform_id: str = "diff1",
    dataset_version: str | None = None,
    start: str | None = None,
    end: str | None = None,
    write_manifest_files: bool = True,
) -> None:
    series_id = series_id.strip().upper()
    transform_id = transform_id.strip().lower()
    series_slug = _series_slug(series_id)

    TARGETS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = fetch_series(series_id=series_id, start=start, end=end)
    transformed_df = apply_transform(raw_df, transform_id=transform_id)
    if transformed_df.empty:
        raise RuntimeError(
            f"Transformed series is empty for series_id={series_id!r}, transform={transform_id!r}."
        )

    raw_col = series_slug
    transformed_col = f"{series_slug}_{transform_id}"
    raw_out = raw_df.rename(columns={"value": raw_col})
    transformed_out = transformed_df.rename(columns={"value_transformed": transformed_col})

    raw_path = TARGETS_DIR / f"{series_slug}_raw.parquet"
    transformed_path = TARGETS_DIR / f"{series_slug}_{transform_id}.parquet"
    _atomic_write_parquet(raw_out, raw_path)
    _atomic_write_parquet(transformed_out, transformed_path)
    logger.info("Wrote raw target series: %s (%d rows)", raw_path, len(raw_out))
    logger.info("Wrote transformed target series: %s (%d rows)", transformed_path, len(transformed_out))

    if write_manifest_files:
        manifest = TargetDatasetManifest(
            dataset_version=dataset_version
            or default_dataset_version(prefix=f"targets-{series_slug}"),
            created_at_utc=utc_now_iso(),
            git_sha=get_git_sha(REPO_ROOT),
            series_id=series_id,
            transform_id=transform_id,
            raw_output_path=str(raw_path),
            raw_output_sha256=hash_file(raw_path),
            raw_output_rows=len(raw_out),
            transformed_output_path=str(transformed_path),
            transformed_output_sha256=hash_file(transformed_path),
            transformed_output_rows=len(transformed_out),
            fetch_start=start,
            fetch_end=end,
        )

        manifest_path = write_manifest(
            out_dir=TARGETS_DIR,
            registry_path=TARGETS_DIR / _MANIFEST_REGISTRY,
            manifest=manifest,
            extra={"fred_api_key_env": "FRED_API_KEY"},
        )
        logger.info(
            "Wrote target manifest %s (dataset_version=%s)",
            manifest_path,
            manifest.dataset_version,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and transform target series from FRED.")
    p.add_argument(
        "--series-id",
        type=str,
        default="T5YIE",
        help="FRED series identifier (default: T5YIE).",
    )
    p.add_argument(
        "--transform",
        type=str,
        default="diff1",
        choices=list(_SUPPORTED_TRANSFORMS),
        help="Transformation id to apply (default: diff1).",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) for FRED fetch window.",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) for FRED fetch window.",
    )
    p.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Optional explicit dataset version (default: autogenerated UTC tag).",
    )
    p.add_argument(
        "--manifest",
        dest="write_manifest_files",
        action="store_true",
        default=True,
        help="Write manifest JSON + dataset_registry row (default: enabled).",
    )
    p.add_argument(
        "--no-manifest",
        dest="write_manifest_files",
        action="store_false",
        help="Disable manifest/registry writes.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        series_id=args.series_id,
        transform_id=args.transform,
        dataset_version=args.dataset_version,
        start=args.start,
        end=args.end,
        write_manifest_files=args.write_manifest_files,
    )

