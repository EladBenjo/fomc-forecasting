"""
Discovery stage for FOMC policy documents.

The Fed's materials page is Angular-rendered but exposes clean JSON endpoints
that back the UI. We read those directly — no HTML scraping needed.

JSON sources:
  final-recent.json  — rolling window of recent meetings
  final-hist.json    — historical archive (2010+)

Supported categories (type codes):
  St   — Policy Statements
  Mn   — Minutes
  PrC  — Press Conference transcripts
"""

import json
import logging
import sqlite3
from datetime import date

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalreserve.gov"
MATERIALS_BASE = BASE_URL + "/monetarypolicy/materials/assets/"

JSON_FEEDS = [
    MATERIALS_BASE + "final-recent.json",
    MATERIALS_BASE + "final-hist.json",
]

# Categories to ingest — extend this list to add more
DEFAULT_CATEGORIES = {"St", "Mn"}


def _fetch_json(session: requests.Session, url: str) -> list[dict]:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = json.loads(resp.content.decode("utf-8-sig"))
    return data.get("mtgitems", [])


def _extract_urls(item: dict) -> tuple[str | None, str | None]:
    """Return (html_url, pdf_url) from a document item."""
    # Some items have a single 'url'; others have a 'files' list
    if "files" in item:
        def _label(f: dict) -> str:
            return f.get("name") or f.get("link") or ""

        html_url = next(
            (BASE_URL + f["url"] for f in item["files"] if _label(f) == "HTML"), None
        )
        pdf_url = next(
            (BASE_URL + f["url"] for f in item["files"] if _label(f) == "PDF"), None
        )
    elif "url" in item:
        url = BASE_URL + item["url"]
        html_url = url if not url.endswith(".pdf") else None
        pdf_url  = url if url.endswith(".pdf") else None
    else:
        html_url = pdf_url = None
    return html_url, pdf_url


def _make_doc_id(item: dict) -> str:
    """Stable unique ID: category + publication date (falls back to meeting date)."""
    pub_date = item.get("dt") or item.get("d", "unknown")
    return f"{item['type']}_{pub_date.replace('-', '')}"


def _save_document(conn: sqlite3.Connection, doc: dict) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO documents
            (doc_id, category, meeting_date, pub_date, meeting_label,
             html_url, pdf_url, scrape_date)
        VALUES
            (:doc_id, :category, :meeting_date, :pub_date, :meeting_label,
             :html_url, :pdf_url, :scrape_date)
        """,
        {**doc, "scrape_date": date.today().isoformat()},
    )
    conn.commit()


def run(
    conn: sqlite3.Connection,
    categories: set[str] = DEFAULT_CATEGORIES,
) -> None:
    """Crawl FOMC JSON feeds and populate the documents table."""
    session = requests.Session()
    session.headers["User-Agent"] = "fedtext-scraper/1.0 (research)"

    seen: set[str] = set()

    for feed_url in JSON_FEEDS:
        logger.info("Fetching feed: %s", feed_url)
        try:
            items = _fetch_json(session, feed_url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", feed_url, exc)
            continue

        matched = [i for i in items if i.get("type") in categories]
        logger.info("  %d items match categories %s", len(matched), categories)

        for item in matched:
            doc_id = _make_doc_id(item)
            if doc_id in seen:
                continue  # recent + hist feeds overlap; skip duplicates
            seen.add(doc_id)

            html_url, pdf_url = _extract_urls(item)
            doc = {
                "doc_id":        doc_id,
                "category":      item["type"],
                "meeting_date":  item["d"],
                "pub_date":      item.get("dt") or item.get("d"),
                "meeting_label": item.get("mtg", ""),
                "html_url":      html_url,
                "pdf_url":       pdf_url,
            }
            _save_document(conn, doc)

    total = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE category IN ({})".format(
            ",".join("?" * len(categories))
        ),
        list(categories),
    ).fetchone()[0]
    logger.info("Documents table now has %d rows for %s", total, categories)
