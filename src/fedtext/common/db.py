import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parents[3] / "data" / "catalog" / "speeches.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_speeches_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS speeches (
            id           INTEGER PRIMARY KEY,
            speech_date  DATE,
            title        TEXT,
            speaker      TEXT,
            event        TEXT,
            link         TEXT UNIQUE,
            speech_text  TEXT,
            scrape_date  DATE,
            processed    BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_speech_date ON speeches(speech_date);
        CREATE INDEX IF NOT EXISTS idx_speaker     ON speeches(speaker);
        CREATE INDEX IF NOT EXISTS idx_processed   ON speeches(processed);
    """)
    conn.commit()
