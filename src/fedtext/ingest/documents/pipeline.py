"""
FOMC documents ingestion pipeline entry point.

Usage:
    python -m fedtext.ingest.documents.pipeline
    python -m fedtext.ingest.documents.pipeline --categories St Mn
    python -m fedtext.ingest.documents.pipeline --discovery-only
    python -m fedtext.ingest.documents.pipeline --fetch-only
    python -m fedtext.ingest.documents.pipeline --parse-only
    python -m fedtext.ingest.documents.pipeline --limit 5
"""

import argparse
import logging
import sys

from fedtext.common.db import get_documents_connection, init_documents_db
from fedtext.ingest.documents.discovery import crawler
from fedtext.ingest.documents.fetch import downloader
from fedtext.ingest.documents.parse import parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run(
    categories: set[str] = frozenset({"St", "Mn"}),
    discovery_only: bool = False,
    fetch_only: bool = False,
    parse_only: bool = False,
    limit: int | None = None,
) -> None:
    conn = get_documents_connection()
    init_documents_db(conn)

    if not fetch_only and not parse_only:
        logger.info("=== DISCOVERY  categories=%s ===", categories)
        crawler.run(conn, categories=categories)

    if not discovery_only and not parse_only:
        logger.info("=== FETCH ===")
        downloader.run(conn, limit=limit)

    if not discovery_only and not fetch_only:
        logger.info("=== PARSE ===")
        parser.run(conn)

    conn.close()
    logger.info("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest FOMC policy documents")
    p.add_argument("--categories", nargs="+", default=["St", "Mn"],
                   help="Document type codes to ingest (default: St Mn)")
    p.add_argument("--discovery-only", action="store_true")
    p.add_argument("--fetch-only",     action="store_true")
    p.add_argument("--parse-only",     action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Max documents to fetch (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        categories=set(args.categories),
        discovery_only=args.discovery_only,
        fetch_only=args.fetch_only,
        parse_only=args.parse_only,
        limit=args.limit,
    )
