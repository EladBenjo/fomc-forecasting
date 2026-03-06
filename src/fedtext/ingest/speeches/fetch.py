"""
Fetch stage for Fed speeches.

For each speech in the DB that hasn't been processed yet, fetches the
individual speech page, extracts the body text, and writes it back to the DB.

Two HTML layouts are handled:
  - 2006+  : article body in  #article > div:nth-child(3)
  - pre-2006: keyword-delimited full-page text extraction
"""

import logging
import re
import sqlite3
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FETCH_DELAY = 2.0  # seconds — be polite to the Fed's servers

# Pre-2006 pages have a "Return to top" sentinel that marks where speech text begins
_START_KEYWORD = "Return to top"
_END_KEYWORDS  = ["Footnotes", "References", "Endnotes"]
_NOISE_PHRASES = ["Return to top", "Watch live", "Return to text"]


def _extract_text_new(soup: BeautifulSoup) -> str | None:
    """Extract text from 2006+ speech pages."""
    content_div = soup.select_one("#article > div:nth-child(3)")
    if not content_div:
        return None
    text = content_div.get_text()
    # Trim at the first footnote/endnote section
    end_idx = len(text)
    for kw in _END_KEYWORDS:
        idx = text.find(kw)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    return text[:end_idx].strip()


def _extract_text_old(soup: BeautifulSoup) -> str | None:
    """Extract text from pre-2006 speech pages using sentinel keywords."""
    full_text = soup.get_text()
    first_idx = full_text.find(_START_KEYWORD)
    if first_idx == -1:
        return None
    second_idx = full_text.find(_START_KEYWORD, first_idx + len(_START_KEYWORD))
    if second_idx == -1:
        return None
    start_idx = second_idx + len(_START_KEYWORD)

    end_idx = len(full_text)
    for kw in _END_KEYWORDS:
        idx = full_text.find(kw, start_idx)
        if idx != -1 and idx < end_idx:
            end_idx = idx

    return full_text[start_idx:end_idx].strip()


def _clean(text: str) -> str:
    for phrase in _NOISE_PHRASES:
        text = text.replace(phrase, "")
    return re.sub(r"\n{2,}", "\n", text).strip()


def _fetch_speech_text(session: requests.Session, link: str, year: int) -> str | None:
    try:
        resp = session.get(link, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", link, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    raw = _extract_text_new(soup) if year >= 2006 else _extract_text_old(soup)
    if raw is None:
        logger.warning("Could not extract text from %s", link)
        return None
    return _clean(raw)


def run(conn: sqlite3.Connection) -> None:
    """Fetch and extract text for all unprocessed speeches."""
    rows = conn.execute(
        "SELECT id, link, speech_date FROM speeches WHERE processed = FALSE ORDER BY speech_date"
    ).fetchall()

    if not rows:
        logger.info("No unprocessed speeches found.")
        return

    logger.info("Fetching text for %d speeches...", len(rows))
    session = requests.Session()
    session.headers["User-Agent"] = "fedtext-scraper/1.0 (research)"

    for row in rows:
        speech_id = row["id"]
        link      = row["link"]
        year      = int(str(row["speech_date"])[:4])

        logger.info("Processing speech id=%d  %s", speech_id, link)
        text = _fetch_speech_text(session, link, year)

        if text:
            conn.execute(
                "UPDATE speeches SET speech_text = ?, processed = TRUE WHERE id = ?",
                (text, speech_id),
            )
            conn.commit()
            logger.info("  -> saved %d chars", len(text))
        else:
            logger.warning("  -> skipped (no text extracted)")

        time.sleep(FETCH_DELAY)
