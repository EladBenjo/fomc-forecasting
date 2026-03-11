-- Migration 003: add chunks table (applies to both databases)
-- Run against speeches.db  with source_type default 'speech'
-- Run against catalog.sqlite with source_type default 'document'

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
