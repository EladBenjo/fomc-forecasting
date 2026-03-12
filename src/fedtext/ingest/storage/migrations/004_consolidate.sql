-- Migration 004: unified fedtext.db schema
-- Combines speeches and documents into a single database.
-- Run by scripts/consolidate_dbs.py (one-time migration).

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

CREATE INDEX IF NOT EXISTS idx_doc_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_doc_meeting  ON documents(meeting_date);
CREATE INDEX IF NOT EXISTS idx_doc_fetched  ON documents(fetched);

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
    source_type TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    token_est   INTEGER,
    embedding   BLOB,
    UNIQUE(source_type, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunk_doc    ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunks(source_type, source_id);
