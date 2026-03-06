"""
Discovery stage for Fed speeches.

Crawls the Federal Reserve speech listing pages (1996–present) and saves
speech metadata (title, speaker, event, link, date) to the speeches table.
Two URL formats and two HTML layouts are handled, matching the source site's
historical structure.
"""

import logging
import re
import sqlite3
import time
from datetime import datetime, date

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalreserve.gov"

# The Fed switched URL formats and HTML layouts around 2006/2011
URL_FORMAT_OLD = BASE_URL + "/newsevents/speech/{year}speech.htm"   # 1996–2010
URL_FORMAT_NEW = BASE_URL + "/newsevents/speech/{year}-speeches.htm" # 2011–present

CRAWL_DELAY = 1.5  # seconds between requests


def _listing_url(year: int) -> str:
    return URL_FORMAT_OLD.format(year=year) if year <= 2010 else URL_FORMAT_NEW.format(year=year)


def _parse_date_from_url(url: str, fallback_year: int) -> str:
    """Extract date from the 8-digit sequence embedded in Fed speech URLs."""
    match = re.search(r"\d{8}", url)
    if match:
        return datetime.strptime(match.group(0), "%Y%m%d").strftime("%Y-%m-%d")
    logger.warning("No date found in URL %s; defaulting to Jan 1 %s", url, fallback_year)
    return date(fallback_year, 1, 1).isoformat()


def _parse_listing_page(soup: BeautifulSoup, year: int) -> list[dict]:
    """Return a list of speech metadata dicts from a parsed listing page."""
    speeches = []

    if year < 2006:
        titles    = soup.select(".title")
        speakers  = soup.select(".speaker")
        locations = soup.select(".location")

        for i in range(min(len(titles), len(speakers), len(locations))):
            anchor = titles[i].find("a", href=True)
            if not anchor:
                continue
            link = BASE_URL + anchor["href"]
            speeches.append({
                "speech_date": _parse_date_from_url(link, year),
                "link":        link,
                "title":       titles[i].text.strip(),
                "speaker":     speakers[i].text.strip(),
                "event":       locations[i].text.strip(),
            })
    else:
        for event in soup.select(".eventlist__event"):
            anchor = event.find("a", href=True)
            if not anchor:
                continue
            link = BASE_URL + anchor["href"]
            parts = [p.strip() for p in event.text.split("\n") if p.strip()]
            if len(parts) < 2:
                continue
            # Skip video-only entries
            speaker_idx = 1 if parts[1] not in {"Watch Live", "Video"} else 2
            if speaker_idx >= len(parts):
                continue
            speeches.append({
                "speech_date": _parse_date_from_url(link, year),
                "link":        link,
                "title":       parts[0],
                "speaker":     parts[speaker_idx],
                "event":       parts[speaker_idx + 1] if speaker_idx + 1 < len(parts) else "",
            })

    return speeches


def _save_speech(conn: sqlite3.Connection, speech: dict) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO speeches (speech_date, title, speaker, event, link, scrape_date)
        VALUES (:speech_date, :title, :speaker, :event, :link, :scrape_date)
        """,
        {**speech, "scrape_date": date.today().isoformat()},
    )
    conn.commit()


def run(conn: sqlite3.Connection, start_year: int = 1996, end_year: int | None = None) -> None:
    """Crawl speech listing pages and populate the speeches table."""
    if end_year is None:
        end_year = datetime.now().year

    session = requests.Session()
    session.headers["User-Agent"] = "fedtext-scraper/1.0 (research)"

    for year in range(start_year, end_year + 1):
        url = _listing_url(year)
        logger.info("Fetching listing for %d: %s", year, url)
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            time.sleep(CRAWL_DELAY)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        speeches = _parse_listing_page(soup, year)
        logger.info("Found %d speeches for %d", len(speeches), year)

        for speech in speeches:
            _save_speech(conn, speech)

        time.sleep(CRAWL_DELAY)
