"""
Fetch stage for Fed speeches.

For each speech in the DB that hasn't been processed yet, fetches the
individual speech page, extracts the body text, and writes it back to the DB.

Two HTML layouts are handled:
  - 2006+  : article body in #article > div:nth-child(3), with article fallbacks
  - pre-2006: year-delimited full-page text extraction
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FETCH_DELAY = 2.0  # seconds - be polite to the Fed's servers

_END_KEYWORDS = ["Footnotes", "References", "Endnotes"]
_NOISE_PHRASES = {
    "PDF",
    "Share",
    "Watch Live",
    "Watch live",
    "Return to top",
    "Return to text",
    "Please enable JavaScript if it is disabled in your browser or access the information through the links provided below.",
}
_MIN_TEXT_CHARS = 80


def _trim_at_first_marker(text: str, markers: list[str], *, start_idx: int = 0) -> str:
    end_idx = len(text)
    for marker in markers:
        idx = text.find(marker, start_idx)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    return text[start_idx:end_idx]


def _is_useful_text(text: str) -> bool:
    return len(text) >= _MIN_TEXT_CHARS and len(text.split()) >= 10


def _node_text(node) -> str:
    return node.get_text("\n", strip=True)


def _clean(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line in _NOISE_PHRASES:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_text_new(soup: BeautifulSoup) -> str | None:
    """Extract text from 2006+ speech pages."""
    candidates: list[str] = []
    fallback_candidates: list[str] = []

    content_div = soup.select_one("#article > div:nth-child(3)")
    if content_div is not None:
        candidates.append(_node_text(content_div))

    article = soup.select_one("#article")
    if article is not None:
        candidates.append(_node_text(article))
        fallback_candidates.extend(
            _node_text(node) for node in article.find_all(["div", "section"], recursive=False)
        )

    fallback_candidates.extend(_node_text(node) for node in soup.find_all(["article", "main"]))

    seen: set[str] = set()
    ordered_candidates = candidates + sorted(set(fallback_candidates), key=len, reverse=True)
    for text in ordered_candidates:
        if text in seen:
            continue
        seen.add(text)
        cleaned = _clean(_trim_at_first_marker(text, _END_KEYWORDS))
        if _is_useful_text(cleaned):
            return cleaned
    return None


def _extract_text_old(soup: BeautifulSoup, year: int) -> str | None:
    """Extract text from pre-2006 speech pages using the notebook-derived year marker."""
    full_text = soup.get_text()
    year_marker = str(year)
    first_idx = full_text.find(year_marker)
    if first_idx == -1:
        return None

    second_idx = full_text.find(year_marker, first_idx + len(year_marker))
    start_idx = (second_idx if second_idx != -1 else first_idx) + len(year_marker)
    text = _trim_at_first_marker(
        full_text,
        [f"{year_marker} Speeches", *_END_KEYWORDS],
        start_idx=start_idx,
    )
    cleaned = _clean(text)
    return cleaned if _is_useful_text(cleaned) else None


def _fetch_speech_text(session: requests.Session, link: str, year: int) -> str | None:
    try:
        resp = session.get(link, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", link, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    raw = _extract_text_new(soup) if year >= 2006 else _extract_text_old(soup, year)
    if raw is None:
        logger.warning("Could not extract text from %s", link)
        return None
    return raw


def run(
    conn: sqlite3.Connection,
    limit: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    """Fetch and extract text for all unprocessed speeches."""
    conditions = ["processed = FALSE"]
    params: list[int] = []
    if start_year is not None:
        conditions.append("CAST(substr(speech_date, 1, 4) AS INTEGER) >= ?")
        params.append(int(start_year))
    if end_year is not None:
        conditions.append("CAST(substr(speech_date, 1, 4) AS INTEGER) <= ?")
        params.append(int(end_year))

    query = (
        "SELECT id, link, speech_date FROM speeches "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY speech_date"
    )
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(query, params).fetchall()

    if not rows:
        logger.info("No unprocessed speeches found.")
        return

    logger.info("Fetching text for %d speeches...", len(rows))
    session = requests.Session()
    session.headers["User-Agent"] = "fedtext-scraper/1.0 (research)"

    for row in rows:
        speech_id = row["id"]
        link = row["link"]
        year = int(str(row["speech_date"])[:4])

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
