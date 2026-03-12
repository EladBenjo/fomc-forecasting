"""Completeness checks for ingested data.

Flags documents and speeches that are missing key fields or have not
progressed through the expected pipeline stages. Results are printed as
a structured report and returned as a dict for downstream use (e.g. the
Streamlit status page).

Usage:
    python -m fedtext.ingest.validators.completeness
    python -m fedtext.ingest.validators.completeness --db documents
    python -m fedtext.ingest.validators.completeness --db speeches
"""

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass, field

from fedtext.common.db import get_connection

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    severity: str   # "error" | "warning"
    table: str
    check: str
    count: int
    detail: str = ""


@dataclass
class ValidationReport:
    db_label: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def print_summary(self) -> None:
        print(f"\n=== Completeness report: {self.db_label} ===")
        if not self.issues:
            print("  All checks passed.")
            return
        for issue in self.issues:
            icon = "ERROR" if issue.severity == "error" else "WARN "
            msg = f"  [{icon}] {issue.table}.{issue.check}: {issue.count} row(s)"
            if issue.detail:
                msg += f" — {issue.detail}"
            print(msg)
        print(f"  Totals: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")


# ---------------------------------------------------------------------------
# Documents checks
# ---------------------------------------------------------------------------

def _check_documents(conn: sqlite3.Connection) -> list[Issue]:
    issues: list[Issue] = []

    checks = [
        ("error",   "missing_meeting_date",
         "SELECT COUNT(*) FROM documents WHERE meeting_date IS NULL OR meeting_date = ''",
         "documents have no meeting_date"),
        ("error",   "missing_html_and_pdf_url",
         "SELECT COUNT(*) FROM documents WHERE html_url IS NULL AND pdf_url IS NULL",
         "documents have neither html_url nor pdf_url"),
        ("warning", "not_fetched",
         "SELECT COUNT(*) FROM documents WHERE fetched = FALSE",
         "documents discovered but not yet fetched"),
        ("warning", "fetched_not_parsed",
         "SELECT COUNT(*) FROM documents WHERE fetched = TRUE AND parsed = FALSE",
         "documents fetched but not parsed"),
        ("error",   "parsed_empty_text",
         "SELECT COUNT(*) FROM documents WHERE parsed = TRUE AND (doc_text IS NULL OR TRIM(doc_text) = '')",
         "parsed documents with empty text"),
    ]

    for severity, check_name, sql, detail in checks:
        count = conn.execute(sql).fetchone()[0]
        if count:
            issues.append(Issue(severity, "documents", check_name, count, detail))

    return issues


# ---------------------------------------------------------------------------
# Speeches checks
# ---------------------------------------------------------------------------

def _check_speeches(conn: sqlite3.Connection) -> list[Issue]:
    issues: list[Issue] = []

    checks = [
        ("error",   "missing_speech_date",
         "SELECT COUNT(*) FROM speeches WHERE speech_date IS NULL OR speech_date = ''",
         "speeches have no speech_date"),
        ("error",   "missing_link",
         "SELECT COUNT(*) FROM speeches WHERE link IS NULL OR link = ''",
         "speeches have no link"),
        ("warning", "no_text_fetched",
         "SELECT COUNT(*) FROM speeches WHERE speech_text IS NULL OR TRIM(speech_text) = ''",
         "speeches with no fetched text"),
        ("error",   "missing_speaker",
         "SELECT COUNT(*) FROM speeches WHERE speaker IS NULL OR TRIM(speaker) = ''",
         "speeches missing speaker name"),
    ]

    for severity, check_name, sql, detail in checks:
        count = conn.execute(sql).fetchone()[0]
        if count:
            issues.append(Issue(severity, "speeches", check_name, count, detail))

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_documents() -> ValidationReport:
    """Run completeness checks on the documents table."""
    conn = get_connection()
    report = ValidationReport(db_label="documents (fedtext.db)")
    report.issues = _check_documents(conn)
    conn.close()
    return report


def validate_speeches() -> ValidationReport:
    """Run completeness checks on the speeches table."""
    conn = get_connection()
    report = ValidationReport(db_label="speeches (fedtext.db)")
    report.issues = _check_speeches(conn)
    conn.close()
    return report


def validate_all() -> list[ValidationReport]:
    return [validate_documents(), validate_speeches()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run completeness checks on ingested data")
    p.add_argument(
        "--db",
        choices=["documents", "speeches", "all"],
        default="all",
        help="Which database to validate (default: all)",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(sys.stdout)])
    args = _parse_args()

    if args.db == "documents":
        reports = [validate_documents()]
    elif args.db == "speeches":
        reports = [validate_speeches()]
    else:
        reports = validate_all()

    any_errors = False
    for report in reports:
        report.print_summary()
        if report.errors:
            any_errors = True

    sys.exit(1 if any_errors else 0)
