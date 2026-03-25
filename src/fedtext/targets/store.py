"""SQLite storage helpers for TS + text-feature notebook workflows.

This module materializes a notebook-ready SQLite database that combines:
  - document-level text features from features.parquet
  - target raw series parquet
  - target transformed series parquet

Target artifacts are sourced from ``data/targets`` and can be auto-fetched
through ``fedtext.targets.pipeline`` when missing.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fedtext.common.paths import TARGETS_DIR
import fedtext.targets.pipeline as targets_pipeline

logger = logging.getLogger(__name__)


def _series_slug(series_id: str) -> str:
    return series_id.strip().lower()


def _target_paths(
    *,
    targets_dir: Path,
    series_id: str,
    transform_id: str,
) -> tuple[Path, Path]:
    slug = _series_slug(series_id)
    transform_slug = transform_id.strip().lower()
    raw_path = targets_dir / f"{slug}_raw.parquet"
    transformed_path = targets_dir / f"{slug}_{transform_slug}.parquet"
    return raw_path, transformed_path


def _normalize_date(df: pd.DataFrame, *, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns:
        raise ValueError(f"Expected '{column}' column.")
    out[column] = pd.to_datetime(out[column], errors="coerce")
    out = out.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
    out[column] = out[column].dt.strftime("%Y-%m-%d")
    return out


def _ensure_target_artifacts(
    *,
    targets_dir: Path,
    series_id: str,
    transform_id: str,
    auto_fetch_missing: bool,
    start: str | None,
    end: str | None,
) -> tuple[Path, Path]:
    raw_path, transformed_path = _target_paths(
        targets_dir=targets_dir,
        series_id=series_id,
        transform_id=transform_id,
    )
    if raw_path.exists() and transformed_path.exists():
        return raw_path, transformed_path

    if not auto_fetch_missing:
        raise FileNotFoundError(
            "Missing target artifacts. Run "
            f"`python -m fedtext.targets.pipeline --series-id {series_id} --transform {transform_id}` "
            "or re-run with auto_fetch_missing=True."
        )

    logger.info(
        "Target parquet missing; auto-fetching via fedtext.targets.pipeline (series=%s, transform=%s).",
        series_id,
        transform_id,
    )
    targets_pipeline.run(
        series_id=series_id,
        transform_id=transform_id,
        start=start,
        end=end,
        write_manifest_files=True,
    )
    if raw_path.exists() and transformed_path.exists():
        return raw_path, transformed_path

    raise RuntimeError(
        "Target pipeline completed but expected parquet outputs were not found: "
        f"{raw_path}, {transformed_path}"
    )


def build_eda_db(
    db_path: str | Path,
    features_parquet: str | Path,
    *,
    series_id: str = "T5YIE",
    transform_id: str = "diff1",
    auto_fetch_missing: bool = True,
    start: str | None = None,
    end: str | None = None,
    targets_dir: str | Path | None = None,
) -> Path:
    """Build a notebook-ready SQLite DB for TS + text EDA.

    Tables written (replace semantics for idempotent reruns):
      - ``features_doc_level``
      - ``target_raw``
      - ``target_transformed``
      - ``build_metadata``
    """
    db_path = Path(db_path)
    features_path = Path(features_parquet)
    effective_targets_dir = Path(targets_dir) if targets_dir is not None else TARGETS_DIR

    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {features_path}")

    raw_path, transformed_path = _ensure_target_artifacts(
        targets_dir=effective_targets_dir,
        series_id=series_id,
        transform_id=transform_id,
        auto_fetch_missing=auto_fetch_missing,
        start=start,
        end=end,
    )

    features_df = _normalize_date(pd.read_parquet(features_path), column="date")
    raw_df = _normalize_date(pd.read_parquet(raw_path), column="date")
    transformed_df = _normalize_date(pd.read_parquet(transformed_path), column="date")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        features_df.to_sql("features_doc_level", conn, if_exists="replace", index=False)
        raw_df.to_sql("target_raw", conn, if_exists="replace", index=False)
        transformed_df.to_sql("target_transformed", conn, if_exists="replace", index=False)

        metadata = pd.DataFrame(
            [
                {
                    "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "series_id": series_id.strip().upper(),
                    "transform_id": transform_id.strip().lower(),
                    "features_parquet_path": str(features_path),
                    "target_raw_parquet_path": str(raw_path),
                    "target_transformed_parquet_path": str(transformed_path),
                    "auto_fetch_missing": int(bool(auto_fetch_missing)),
                    "fetch_start": start,
                    "fetch_end": end,
                }
            ]
        )
        metadata.to_sql("build_metadata", conn, if_exists="replace", index=False)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_doc_level_date ON features_doc_level(date)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_target_raw_date ON target_raw(date)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_target_transformed_date ON target_transformed(date)"
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("Wrote EDA SQLite DB: %s", db_path)
    return db_path


__all__ = ["build_eda_db"]
