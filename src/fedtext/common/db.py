import sqlite3
from pathlib import Path

from fedtext.common.paths import FEDTEXT_DB

_MIGRATION_SQL = (
    Path(__file__).parents[2]
    / "fedtext" / "ingest" / "storage" / "migrations" / "004_consolidate.sql"
)


def get_connection(db_path: Path = FEDTEXT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialise all tables in the unified fedtext.db."""
    conn.executescript(_MIGRATION_SQL.read_text())
    conn.commit()
