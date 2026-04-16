"""
Feature engineering pipeline entry point.

Reads parsed text from fedtext.db, computes sentiment + novelty, and writes
a parquet file to data/features/doc_level/features.parquet.

Usage:
    python -m fedtext.text.features.pipeline
    python -m fedtext.text.features.pipeline --source-types speeches documents
    python -m fedtext.text.features.pipeline --limit 50
    python -m fedtext.text.features.pipeline --checkpoint-every 25
    python -m fedtext.text.features.pipeline --no-resume
    python -m fedtext.text.features.pipeline --reset-checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fedtext.common.db import get_connection
from fedtext.common.paths import FEATURES_DIR, FEDTEXT_DB, REPO_ROOT
from fedtext.text.cleaning.normalizer import normalize, split_sentences
from fedtext.text.features import document_features as doc_features
from fedtext.text.features import novelty as novelty_mod
from fedtext.text.features import sentiment as sentiment_mod
from fedtext.text.features.versioning import (
    DatasetManifest,
    default_dataset_version,
    get_git_sha,
    hash_file,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_OUT_DIR = FEATURES_DIR / "doc_level"
_STATE_DB = _OUT_DIR / "features_state.sqlite3"
_CACHE_DB = _OUT_DIR / "features_cache.sqlite3"
_REGISTRY_DB = _OUT_DIR / "dataset_registry.sqlite3"
_CHECKPOINT_TABLE = "features_doc_checkpoint"
_OUTPUT_COLUMNS = [
    "doc_id",
    "source_type",
    "date",
    "hawkish_score",
    "n_hawkish",
    "n_dovish",
    "n_neutral",
    "n_target_sentences",
    "text_length_words",
    "role",
    "target_sentences_ratio",
    "novelty",
]
_SOURCE_TYPE_TO_ROW_LABEL = {
    "speeches": "speech",
    "documents": "document",
    "speech": "speech",
    "document": "document",
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_speeches(conn, limit: int | None) -> list[dict]:
    sql = """
        SELECT
            id          AS doc_id,
            'speech'    AS source_type,
            speech_date AS date,
            speech_text AS text,
            speaker     AS speaker,
            title       AS title,
            event       AS event
        FROM speeches
        WHERE speech_text IS NOT NULL AND speech_text != ''
        ORDER BY speech_date
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
            doc_text        AS text,
            NULL            AS speaker,
            NULL            AS title,
            NULL            AS event
        FROM documents
        WHERE doc_text IS NOT NULL AND doc_text != ''
        ORDER BY meeting_date
    """
    if limit:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def _open_state_conn() -> sqlite3.Connection:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_STATE_DB)
    _init_state_schema(conn)
    return conn


def _to_row_source_types(source_types: list[str]) -> list[str]:
    row_types = [_SOURCE_TYPE_TO_ROW_LABEL[s] for s in source_types]
    # Preserve order while deduplicating.
    return list(dict.fromkeys(row_types))


def _init_state_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CHECKPOINT_TABLE} (
            source_type        TEXT NOT NULL,
            doc_id             INTEGER NOT NULL,
            date               TEXT NOT NULL,
            hawkish_score      REAL NOT NULL,
            n_hawkish          INTEGER NOT NULL,
            n_dovish           INTEGER NOT NULL,
            n_neutral          INTEGER NOT NULL,
            n_target_sentences INTEGER NOT NULL,
            text_length_words  INTEGER,
            role               TEXT,
            target_sentences_ratio REAL,
            novelty            REAL,
            updated_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_type, doc_id)
        )
        """
    )
    _migrate_state_schema(conn)
    conn.commit()


def _checkpoint_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({_CHECKPOINT_TABLE})").fetchall()
    return {str(r[1]) for r in rows}


def _migrate_state_schema(conn: sqlite3.Connection) -> None:
    columns = _checkpoint_columns(conn)
    if "text_length_words" not in columns:
        conn.execute(
            f"ALTER TABLE {_CHECKPOINT_TABLE} ADD COLUMN text_length_words INTEGER"
        )
    if "role" not in columns:
        conn.execute(
            f"ALTER TABLE {_CHECKPOINT_TABLE} ADD COLUMN role TEXT"
        )
    if "target_sentences_ratio" not in columns:
        conn.execute(
            f"ALTER TABLE {_CHECKPOINT_TABLE} ADD COLUMN target_sentences_ratio REAL"
        )


def _reset_checkpoint(conn: sqlite3.Connection, source_types: list[str]) -> None:
    placeholders = ",".join("?" for _ in source_types)
    conn.execute(
        f"DELETE FROM {_CHECKPOINT_TABLE} WHERE source_type IN ({placeholders})",
        source_types,
    )
    conn.commit()


def _load_completed_keys(conn: sqlite3.Connection, source_types: list[str]) -> set[tuple[str, int]]:
    placeholders = ",".join("?" for _ in source_types)
    rows = conn.execute(
        f"""
        SELECT source_type, doc_id
        FROM {_CHECKPOINT_TABLE}
        WHERE source_type IN ({placeholders})
          AND text_length_words IS NOT NULL
          AND target_sentences_ratio IS NOT NULL
        """,
        source_types,
    ).fetchall()
    return {(str(r[0]), int(r[1])) for r in rows}


def _upsert_checkpoint_row(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        f"""
        INSERT INTO {_CHECKPOINT_TABLE} (
            source_type,
            doc_id,
            date,
            hawkish_score,
            n_hawkish,
            n_dovish,
            n_neutral,
            n_target_sentences,
            text_length_words,
            role,
            target_sentences_ratio,
            novelty
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_type, doc_id)
        DO UPDATE SET
            date = excluded.date,
            hawkish_score = excluded.hawkish_score,
            n_hawkish = excluded.n_hawkish,
            n_dovish = excluded.n_dovish,
            n_neutral = excluded.n_neutral,
            n_target_sentences = excluded.n_target_sentences,
            text_length_words = excluded.text_length_words,
            role = excluded.role,
            target_sentences_ratio = excluded.target_sentences_ratio,
            novelty = excluded.novelty,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            row["source_type"],
            int(row["doc_id"]),
            str(row["date"]),
            float(row["hawkish_score"]),
            int(row["n_hawkish"]),
            int(row["n_dovish"]),
            int(row["n_neutral"]),
            int(row["n_target_sentences"]),
            int(row["text_length_words"]),
            row["role"],
            float(row["target_sentences_ratio"]),
            row["novelty"],
        ),
    )


def _load_checkpoint_frame(
    conn: sqlite3.Connection,
    source_types: list[str],
    expected_keys: set[tuple[str, int]],
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in source_types)
    df = pd.read_sql_query(
        f"""
        SELECT
            doc_id,
            source_type,
            date,
            hawkish_score,
            n_hawkish,
            n_dovish,
            n_neutral,
            n_target_sentences,
            text_length_words,
            role,
            target_sentences_ratio,
            novelty
        FROM {_CHECKPOINT_TABLE}
        WHERE source_type IN ({placeholders})
        """,
        conn,
        params=source_types,
    )

    if not expected_keys:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    expected_df = pd.DataFrame(list(expected_keys), columns=["source_type", "doc_id"])
    merged = df.merge(expected_df, on=["source_type", "doc_id"], how="inner")

    if len(merged) != len(expected_keys):
        raise RuntimeError(
            f"Checkpoint state incomplete for selected run: expected {len(expected_keys)} rows, "
            f"found {len(merged)} rows."
        )

    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(["date", "source_type", "doc_id"]).reset_index(drop=True)
    return merged[_OUTPUT_COLUMNS]


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    source_types: list[str] | None = None,
    limit: int | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
    reset_checkpoint: bool = False,
    max_retries: int | None = None,
    dataset_version: str | None = None,
    cleaning_version: str = "1.0.0",
    sentence_split_version: str = "1.0.0",
    write_manifest_files: bool = True,
) -> None:
    if source_types is None:
        source_types = ["speeches", "documents"]
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")

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
        logger.warning("No records found; nothing to do.")
        return

    expected_keys = {(str(r["source_type"]), int(r["doc_id"])) for r in records}
    checkpoint_source_types = _to_row_source_types(source_types)

    # Novelty
    logger.info("Computing novelty scores...")
    novelty_map = novelty_mod.compute_novelty_by_type(records)

    state_conn = _open_state_conn()
    if reset_checkpoint:
        logger.info("Resetting checkpoint rows for selected source types...")
        _reset_checkpoint(state_conn, checkpoint_source_types)

    completed = _load_completed_keys(state_conn, checkpoint_source_types) if resume else set()
    to_process = [
        r for r in records
        if (str(r["source_type"]), int(r["doc_id"])) not in completed
    ]

    if resume:
        logger.info("Resume enabled: %d already checkpointed, %d remaining.", len(completed), len(to_process))

    logger.info("Initializing sentiment client...")
    client = sentiment_mod.load_client(cache_db_path=_CACHE_DB, max_retries=max_retries)

    pending_writes = 0
    try:
        for i, rec in enumerate(to_process):
            if i % 100 == 0:
                logger.info("Sentiment: %d / %d", i, len(to_process))

            text = normalize(rec["text"])
            sents = split_sentences(text)
            result = sentiment_mod.score_document(text, sents, client=client)
            total_sentences_count = len(sents)

            row = {
                "doc_id": rec["doc_id"],
                "source_type": rec["source_type"],
                "date": rec["date"],
                "hawkish_score": result.hawkish_score,
                "n_hawkish": result.n_hawkish,
                "n_dovish": result.n_dovish,
                "n_neutral": result.n_neutral,
                "n_target_sentences": result.n_target_sentences,
                "text_length_words": doc_features.count_words(text),
                "role": doc_features.infer_role(
                    source_type=str(rec["source_type"]),
                    speaker=rec.get("speaker"),
                    title=rec.get("title"),
                    event=rec.get("event"),
                ),
                "target_sentences_ratio": doc_features.compute_target_sentences_ratio(
                    n_target_sentences=result.n_target_sentences,
                    total_sentences_count=total_sentences_count,
                ),
                "novelty": novelty_map.get(rec["doc_id"], float("nan")),
            }
            _upsert_checkpoint_row(state_conn, row)
            pending_writes += 1

            if pending_writes >= checkpoint_every:
                state_conn.commit()
                logger.info("Checkpoint flush: %d docs", checkpoint_every)
                pending_writes = 0
    finally:
        if pending_writes > 0:
            state_conn.commit()
        client.close()

    df = _load_checkpoint_frame(
        conn=state_conn,
        source_types=checkpoint_source_types,
        expected_keys=expected_keys,
    )

    out_path = _OUT_DIR / "features.parquet"
    _atomic_write_parquet(df, out_path)
    logger.info("Wrote %d rows to %s", len(df), out_path)

    state_conn.close()

    if write_manifest_files:
        manifest = DatasetManifest(
            dataset_version=dataset_version or default_dataset_version(),
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            git_sha=get_git_sha(REPO_ROOT),
            output_path=str(out_path),
            output_sha256=hash_file(out_path),
            output_rows=len(df),
            source_types=source_types,
            limit=limit,
            checkpoint_every=checkpoint_every,
            resume=resume,
            reset_checkpoint=reset_checkpoint,
            cleaning_version=cleaning_version,
            sentence_split_version=sentence_split_version,
            input_db_path=str(FEDTEXT_DB),
            input_db_sha256=hash_file(FEDTEXT_DB),
        )
        manifest_path = write_manifest(
            out_dir=_OUT_DIR,
            registry_path=_REGISTRY_DB,
            manifest=manifest,
            extra={"max_retries": max_retries},
        )
        logger.info(
            "Wrote dataset manifest %s (dataset_version=%s)",
            manifest_path,
            manifest.dataset_version,
        )


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
        "--limit", type=int, default=None,
        help="Max documents per source type (for testing)",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=25,
        help="Commit checkpoint progress every N docs (default: 25)",
    )
    p.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Resume from existing checkpoint state (default: enabled)",
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Disable resume and recompute selected docs",
    )
    p.add_argument(
        "--reset-checkpoint", action="store_true",
        help="Delete existing checkpoint rows for selected source types before run",
    )
    p.add_argument(
        "--max-retries", type=int, default=int(os.getenv("ZQ_MAX_RETRIES", "5")),
        help="Max retries for transient API errors (default: env ZQ_MAX_RETRIES or 5)",
    )
    p.add_argument(
        "--dataset-version", type=str, default=None,
        help="Optional explicit dataset version (default: autogenerated UTC timestamp tag)",
    )
    p.add_argument(
        "--cleaning-version", type=str, default=os.getenv("FEDTEXT_CLEANING_VERSION", "1.0.0"),
        help="Semantic version for text cleaning logic (default: 1.0.0)",
    )
    p.add_argument(
        "--sentence-split-version",
        type=str,
        default=os.getenv("FEDTEXT_SENTENCE_SPLIT_VERSION", "1.0.0"),
        help="Semantic version for sentence splitting logic (default: 1.0.0)",
    )
    p.add_argument(
        "--manifest", dest="write_manifest_files", action="store_true", default=True,
        help="Write dataset manifest + registry row (default: enabled)",
    )
    p.add_argument(
        "--no-manifest", dest="write_manifest_files", action="store_false",
        help="Disable writing manifest + registry entry",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        source_types=args.source_types,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        reset_checkpoint=args.reset_checkpoint,
        max_retries=args.max_retries,
        dataset_version=args.dataset_version,
        cleaning_version=args.cleaning_version,
        sentence_split_version=args.sentence_split_version,
        write_manifest_files=args.write_manifest_files,
    )
