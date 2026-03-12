"""One-time migration: merge speeches.db + catalog.sqlite → fedtext.db.

Reads both source databases via SQLite ATTACH and inserts all rows into
the unified fedtext.db. Safe to re-run — uses INSERT OR IGNORE throughout.

Usage:
    python scripts/consolidate_dbs.py
    python scripts/consolidate_dbs.py --dry-run   # print counts, don't write
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Resolve paths relative to repo root
REPO_ROOT    = Path(__file__).parents[1]
SPEECHES_DB  = REPO_ROOT / "data" / "catalog" / "speeches.db"
DOCUMENTS_DB = REPO_ROOT / "data" / "catalog" / "catalog.sqlite"
FEDTEXT_DB   = REPO_ROOT / "data" / "catalog" / "fedtext.db"
MIGRATION_SQL = REPO_ROOT / "src" / "fedtext" / "ingest" / "storage" / "migrations" / "004_consolidate.sql"


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(MIGRATION_SQL.read_text())
    conn.commit()


def _migrate(dry_run: bool = False) -> None:
    if not SPEECHES_DB.exists():
        print(f"ERROR: {SPEECHES_DB} not found", file=sys.stderr)
        sys.exit(1)
    if not DOCUMENTS_DB.exists():
        print(f"ERROR: {DOCUMENTS_DB} not found", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print("[dry-run] No data will be written.\n")

    def _count(conn: sqlite3.Connection, table: str) -> int:
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0  # table doesn't exist in this legacy DB

    # Count source rows
    sp_conn  = sqlite3.connect(SPEECHES_DB)
    doc_conn = sqlite3.connect(DOCUMENTS_DB)
    n_speeches   = _count(sp_conn,  "speeches")
    n_documents  = _count(doc_conn, "documents")
    n_sp_chunks  = _count(sp_conn,  "chunks")
    n_doc_chunks = _count(doc_conn, "chunks")
    sp_conn.close()
    doc_conn.close()

    print(f"Source speeches:        {n_speeches:>6} rows")
    print(f"Source documents:       {n_documents:>6} rows")
    print(f"Source speech chunks:   {n_sp_chunks:>6} rows")
    print(f"Source document chunks: {n_doc_chunks:>6} rows")

    if dry_run:
        return

    FEDTEXT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEDTEXT_DB)
    _init_schema(conn)

    # Migrate via ATTACH
    conn.execute(f"ATTACH DATABASE '{SPEECHES_DB}' AS src_sp")
    conn.execute(f"ATTACH DATABASE '{DOCUMENTS_DB}' AS src_doc")

    def _safe_insert(sql: str) -> None:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # source table didn't exist in legacy DB

    _safe_insert("INSERT OR IGNORE INTO speeches SELECT * FROM src_sp.speeches")
    _safe_insert("INSERT OR IGNORE INTO documents SELECT * FROM src_doc.documents")
    _safe_insert("INSERT OR IGNORE INTO chunks    SELECT * FROM src_sp.chunks")
    _safe_insert("INSERT OR IGNORE INTO chunks    SELECT * FROM src_doc.chunks")

    conn.commit()
    conn.execute("DETACH DATABASE src_sp")
    conn.execute("DETACH DATABASE src_doc")

    # Verify
    n_sp  = conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    n_doc = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_ch  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    print(f"\nfedtext.db speeches:  {n_sp:>6} rows")
    print(f"fedtext.db documents: {n_doc:>6} rows")
    print(f"fedtext.db chunks:    {n_ch:>6} rows")
    print(f"\nWritten to: {FEDTEXT_DB}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge speeches.db + catalog.sqlite → fedtext.db")
    p.add_argument("--dry-run", action="store_true", help="Print counts only, don't write")
    args = p.parse_args()
    _migrate(dry_run=args.dry_run)
