-- Migration 002: initialise documents database
-- Applies to: data/catalog/catalog.sqlite

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
