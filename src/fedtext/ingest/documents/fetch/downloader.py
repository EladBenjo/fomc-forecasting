"""
Fetch stage for FOMC policy documents.

For each discovered document with fetched=FALSE, downloads the HTML page
and stores the raw HTML in the documents table. Prefers HTML over PDF;
PDF support can be added later if needed.
"""

import logging
import sqlite3
import time

import requests

logger = logging.getLogger(__name__)

FETCH_DELAY = 1.5  # seconds


def _fetch_html(session: requests.Session, url: str) -> str | None:
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def run(conn: sqlite3.Connection, limit: int | None = None) -> None:
    """Download HTML for all unfetched documents."""
    query = """
        SELECT id, doc_id, category, html_url, pdf_url
        FROM documents
        WHERE fetched = FALSE AND html_url IS NOT NULL
        ORDER BY meeting_date DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    if not rows:
        logger.info("No unfetched documents found.")
        return

    logger.info("Fetching HTML for %d documents...", len(rows))
    session = requests.Session()
    session.headers["User-Agent"] = "fedtext-scraper/1.0 (research)"

    for row in rows:
        doc_id = row["doc_id"]
        url    = row["html_url"]
        logger.info("Fetching %s  %s", doc_id, url)

        html = _fetch_html(session, url)
        if html:
            conn.execute(
                "UPDATE documents SET fetched = TRUE WHERE id = ?",
                (row["id"],),
            )
            # Store raw HTML temporarily in doc_text; parser will replace it
            conn.execute(
                "UPDATE documents SET doc_text = ? WHERE id = ?",
                (html, row["id"]),
            )
            conn.commit()
            logger.info("  -> %d bytes", len(html))
        else:
            logger.warning("  -> failed")

        time.sleep(FETCH_DELAY)
