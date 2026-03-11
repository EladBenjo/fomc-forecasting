-- Migration 001: initialise speeches database
-- Applies to: data/catalog/speeches.db

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
