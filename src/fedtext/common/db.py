import sqlite3
from pathlib import Path

DB_PATH          = Path(__file__).parents[3] / "data" / "catalog" / "speeches.db"
DOCUMENTS_DB_PATH = Path(__file__).parents[3] / "data" / "catalog" / "catalog.sqlite"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_documents_connection(db_path: Path = DOCUMENTS_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_documents_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id            INTEGER PRIMARY KEY,
            doc_id        TEXT UNIQUE,
            category      TEXT,
            meeting_date  DATE,
            pub_date      DATE,
            meeting_label TEXT,
            html_url      TEXT,
            pdf_url       TEXT,
            doc_text      TEXT,
            fetched       BOOLEAN DEFAULT FALSE,
            parsed        BOOLEAN DEFAULT FALSE,
            scrape_date   DATE
        );

        CREATE INDEX IF NOT EXISTS idx_doc_category  ON documents(category);
        CREATE INDEX IF NOT EXISTS idx_doc_meeting   ON documents(meeting_date);
        CREATE INDEX IF NOT EXISTS idx_doc_fetched   ON documents(fetched);

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY,
            source_id   INTEGER NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'document',
            doc_id      TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT NOT NULL,
            token_est   INTEGER,
            embedding   BLOB,
            UNIQUE(source_type, source_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_doc    ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunks(source_type, source_id);
    """)
    conn.commit()


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

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY,
            source_id   INTEGER NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'speech',
            doc_id      TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT NOT NULL,
            token_est   INTEGER,
            embedding   BLOB,
            UNIQUE(source_type, source_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_doc    ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunks(source_type, source_id);
    """)
    conn.commit()
