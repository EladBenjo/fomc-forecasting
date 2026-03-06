"""
Speech ingestion pipeline entry point.

Usage:
    python -m fedtext.ingest.speeches.pipeline
    python -m fedtext.ingest.speeches.pipeline --start-year 2020
    python -m fedtext.ingest.speeches.pipeline --discovery-only
    python -m fedtext.ingest.speeches.pipeline --fetch-only
"""

import argparse
import logging
import sys
from datetime import datetime

from fedtext.common.db import get_connection, init_speeches_db
from fedtext.ingest.speeches import discovery, fetch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run(
    start_year: int = 1996,
    end_year: int | None = None,
    discovery_only: bool = False,
    fetch_only: bool = False,
) -> None:
    if end_year is None:
        end_year = datetime.now().year

    conn = get_connection()
    init_speeches_db(conn)

    if not fetch_only:
        logger.info("=== DISCOVERY  %d – %d ===", start_year, end_year)
        discovery.run(conn, start_year=start_year, end_year=end_year)

    if not discovery_only:
        logger.info("=== FETCH ===")
        fetch.run(conn)

    conn.close()
    logger.info("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Federal Reserve speeches")
    p.add_argument("--start-year", type=int, default=1996)
    p.add_argument("--end-year",   type=int, default=None)
    p.add_argument("--discovery-only", action="store_true",
                   help="Only crawl listing pages; don't fetch speech text")
    p.add_argument("--fetch-only", action="store_true",
                   help="Only fetch text for already-discovered speeches")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        discovery_only=args.discovery_only,
        fetch_only=args.fetch_only,
    )
