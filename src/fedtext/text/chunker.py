"""
Text chunker for RAG preparation.

Splits a document's text into overlapping paragraph-aligned chunks suitable
for embedding. Each chunk carries its parent's metadata so retrieval results
are fully attributed without extra joins.

Strategy:
  1. Split on blank lines (paragraph boundaries).
  2. Accumulate paragraphs until the chunk would exceed MAX_TOKENS.
  3. Slide forward by STRIDE_TOKENS worth of paragraphs for overlap.

Token estimation: ~4 chars per token (fast, no tokenizer dependency).
"""

import sqlite3
from dataclasses import dataclass

MAX_TOKENS   = 400   # target max tokens per chunk
STRIDE_TOKENS = 100  # overlap between consecutive chunks
CHARS_PER_TOK = 4    # rough estimate


@dataclass
class Chunk:
    chunk_index: int
    chunk_text:  str
    token_est:   int


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOK)


def chunk_text(text: str) -> list[Chunk]:
    """Split `text` into overlapping chunks and return a list of Chunk objects."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    start = 0

    while start < len(paragraphs):
        accumulated: list[str] = []
        tokens = 0
        i = start

        while i < len(paragraphs):
            p_tokens = _estimate_tokens(paragraphs[i])
            if accumulated and tokens + p_tokens > MAX_TOKENS:
                break
            accumulated.append(paragraphs[i])
            tokens += p_tokens
            i += 1

        if not accumulated:
            # Single paragraph exceeds MAX_TOKENS — include it anyway
            accumulated = [paragraphs[start]]
            tokens = _estimate_tokens(accumulated[0])
            i = start + 1

        chunk_text_str = "\n\n".join(accumulated)
        chunks.append(Chunk(
            chunk_index=len(chunks),
            chunk_text=chunk_text_str,
            token_est=tokens,
        ))

        # Advance start by enough paragraphs to cover STRIDE_TOKENS
        stride_consumed = 0
        while start < i and stride_consumed < STRIDE_TOKENS:
            stride_consumed += _estimate_tokens(paragraphs[start])
            start += 1

    return chunks


def run(
    conn: sqlite3.Connection,
    source_type: str,        # 'speech' or 'document'
    source_id: int,
    doc_id: str,
    text: str,
) -> int:
    """Chunk `text` and upsert into the chunks table. Returns number of chunks written."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Delete existing chunks for this source (idempotent re-run)
    conn.execute(
        "DELETE FROM chunks WHERE source_type = ? AND source_id = ?",
        (source_type, source_id),
    )

    conn.executemany(
        """
        INSERT INTO chunks (source_id, source_type, doc_id, chunk_index, chunk_text, token_est)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (source_id, source_type, doc_id, c.chunk_index, c.chunk_text, c.token_est)
            for c in chunks
        ],
    )
    conn.commit()
    return len(chunks)
