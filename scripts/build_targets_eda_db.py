"""Build a notebook-ready SQLite DB for TS + text-feature EDA.

By default this script:
  - reads features from data/features/doc_level/features.parquet
  - reads targets from data/targets/{series}_raw.parquet and {series}_{transform}.parquet
  - auto-fetches target parquet via fedtext.targets.pipeline if missing
  - writes a single SQLite DB under data/targets

Usage:
  python scripts/build_targets_eda_db.py
  python scripts/build_targets_eda_db.py --series-id T5YIE --transform diff1
  python scripts/build_targets_eda_db.py --no-auto-fetch-missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fedtext.common.paths import FEATURES_DIR, TARGETS_DIR
from fedtext.targets.store import build_eda_db


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SQLite DB for TS + text EDA.")
    p.add_argument(
        "--series-id",
        type=str,
        default="T5YIE",
        help="FRED series id (default: T5YIE).",
    )
    p.add_argument(
        "--transform",
        type=str,
        default="diff1",
        help="Target transform id (default: diff1).",
    )
    p.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Output SQLite DB path (default: data/targets/<series>_eda.sqlite3).",
    )
    p.add_argument(
        "--features-parquet",
        type=str,
        default=str(FEATURES_DIR / "doc_level" / "features.parquet"),
        help="Path to doc-level features parquet.",
    )
    p.add_argument(
        "--targets-dir",
        type=str,
        default=str(TARGETS_DIR),
        help="Directory containing target parquet artifacts.",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional FRED fetch start date (YYYY-MM-DD) if auto-fetch is needed.",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional FRED fetch end date (YYYY-MM-DD) if auto-fetch is needed.",
    )
    p.add_argument(
        "--auto-fetch-missing",
        dest="auto_fetch_missing",
        action="store_true",
        default=True,
        help="Auto-fetch missing target parquet via fedtext.targets.pipeline (default: enabled).",
    )
    p.add_argument(
        "--no-auto-fetch-missing",
        dest="auto_fetch_missing",
        action="store_false",
        help="Fail fast if target parquet artifacts are missing.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    slug = args.series_id.strip().lower()
    db_path = Path(args.db_path) if args.db_path else Path(args.targets_dir) / f"{slug}_eda.sqlite3"

    out = build_eda_db(
        db_path=db_path,
        features_parquet=args.features_parquet,
        series_id=args.series_id,
        transform_id=args.transform,
        auto_fetch_missing=args.auto_fetch_missing,
        start=args.start,
        end=args.end,
        targets_dir=args.targets_dir,
    )
    print(f"Wrote EDA DB: {out}")


if __name__ == "__main__":
    main()

