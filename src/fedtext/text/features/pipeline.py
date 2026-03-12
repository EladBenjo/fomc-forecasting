"""
Feature engineering pipeline entry point.

Reads parsed text from fedtext.db, computes sentiment + novelty, and writes
one parquet file per source_type to data/features/doc_level/.

Usage:
    python -m fedtext.text.features.pipeline
    python -m fedtext.text.features.pipeline --source-types speeches documents
    python -m fedtext.text.features.pipeline --device 0   # use GPU
    python -m fedtext.text.features.pipeline --limit 50   # for testing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from fedtext.common.db import get_connection
from fedtext.common.paths import FEATURES_DIR
from fedtext.text.cleaning.normalizer import normalize, split_sentences
from fedtext.text.features import novelty as novelty_mod
from fedtext.text.features import sentiment as sentiment_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_OUT_DIR = FEATURES_DIR / "doc_level"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_speeches(conn, limit: int | None) -> list[dict]:
    sql = """
        SELECT
            id          AS doc_id,
            'speech'    AS source_type,
            date,
            text
        FROM speeches
        WHERE text IS NOT NULL AND text != ''
        ORDER BY date
    """
    if limit:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def _load_documents(conn, limit: int | None) -> list[dict]:
    sql = """
        SELECT
            id              AS doc_id,
            'document'      AS source_type,
            meeting_date    AS date,
            text
        FROM documents
        WHERE text IS NOT NULL AND text != ''
        ORDER BY meeting_date
    """
    if limit:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    source_types: list[str] | None = None,
    device: int = -1,
    limit: int | None = None,
) -> None:
    if source_types is None:
        source_types = ["speeches", "documents"]

    conn = get_connection()

    records: list[dict] = []
    if "speeches" in source_types:
        speeches = _load_speeches(conn, limit)
        logger.info("Loaded %d speeches with text.", len(speeches))
        records.extend(speeches)
    if "documents" in source_types:
        documents = _load_documents(conn, limit)
        logger.info("Loaded %d documents with text.", len(documents))
        records.extend(documents)

    conn.close()

    if not records:
        logger.warning("No records found — nothing to do.")
        return

    # --- Novelty (cheap — no model needed) ---
    logger.info("Computing novelty scores...")
    novelty_map = novelty_mod.compute_novelty_by_type(records)

    # --- Sentiment (expensive — load model once) ---
    logger.info("Loading sentiment model...")
    pipe = sentiment_mod.load_pipeline(device=device)

    rows = []
    for i, rec in enumerate(records):
        if i % 100 == 0:
            logger.info("Sentiment: %d / %d", i, len(records))

        text = normalize(rec["text"])
        sents = split_sentences(text)
        result = sentiment_mod.score_document(text, sents, pipeline=pipe)

        rows.append({
            "doc_id":             rec["doc_id"],
            "source_type":        rec["source_type"],
            "date":               rec["date"],
            "hawkish_score":      result.hawkish_score,
            "n_hawkish":          result.n_hawkish,
            "n_dovish":           result.n_dovish,
            "n_neutral":          result.n_neutral,
            "n_target_sentences": result.n_target_sentences,
            "novelty":            novelty_map.get(rec["doc_id"], float("nan")),
        })

    df = pd.DataFrame(rows)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s", len(df), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute text features for all parsed documents")
    p.add_argument(
        "--source-types", nargs="+",
        default=["speeches", "documents"],
        choices=["speeches", "documents"],
        help="Which tables to process (default: both)",
    )
    p.add_argument(
        "--device", type=int, default=-1,
        help="Torch device: -1=CPU, 0=first GPU (default: -1)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max documents per source type (for testing)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        source_types=args.source_types,
        device=args.device,
        limit=args.limit,
    )
