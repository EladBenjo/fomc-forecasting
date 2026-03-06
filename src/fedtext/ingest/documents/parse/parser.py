"""
Parse stage for FOMC policy documents.

Extracts clean body text from the raw HTML stored in the documents table.

Layout differences by category:
  St  (Statements) — body text is in the 3rd div inside #article
  Mn  (Minutes)    — content is direct children of #article (no wrapper div)
  PrC (Press Conf) — treated like Minutes (direct #article children)
"""

import logging
import re
import sqlite3

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_NOISE_PHRASES = ["Return to top", "Return to text"]
_END_KEYWORDS  = ["Footnotes", "Endnotes", "References"]

# Categories whose body sits in the 3rd div of #article
_DIV3_CATEGORIES = {"St"}


def _clean(text: str) -> str:
    for phrase in _NOISE_PHRASES:
        text = text.replace(phrase, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_statement(soup: BeautifulSoup) -> str | None:
    """Third content div inside #article holds the statement body."""
    article = soup.select_one("#article")
    if not article:
        return None
    divs = [d for d in article.find_all("div", recursive=False) if d.get_text(strip=True)]
    if len(divs) < 2:
        # Fallback: just use all of #article
        return article.get_text()
    # The largest non-empty div is the body
    return max(divs, key=lambda d: len(d.get_text())).get_text()


def _extract_minutes(soup: BeautifulSoup) -> str | None:
    """Minutes content is direct children of #article."""
    article = soup.select_one("#article")
    if not article:
        return None
    text = article.get_text()
    # Trim at footnotes/endnotes
    end_idx = len(text)
    for kw in _END_KEYWORDS:
        idx = text.find(kw)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    return text[:end_idx]


def _parse_html(html: str, category: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    if category in _DIV3_CATEGORIES:
        raw = _extract_statement(soup)
    else:
        raw = _extract_minutes(soup)
    return _clean(raw) if raw else None


def run(conn: sqlite3.Connection) -> None:
    """Parse raw HTML for all fetched-but-not-parsed documents."""
    rows = conn.execute(
        """
        SELECT id, doc_id, category, doc_text
        FROM documents
        WHERE fetched = TRUE AND parsed = FALSE AND doc_text IS NOT NULL
        """
    ).fetchall()

    if not rows:
        logger.info("No unparsed documents found.")
        return

    logger.info("Parsing %d documents...", len(rows))

    for row in rows:
        text = _parse_html(row["doc_text"], row["category"])
        if text:
            conn.execute(
                "UPDATE documents SET doc_text = ?, parsed = TRUE WHERE id = ?",
                (text, row["id"]),
            )
            conn.commit()
            logger.info("  %s -> %d chars", row["doc_id"], len(text))
        else:
            logger.warning("  %s -> could not extract text", row["doc_id"])
