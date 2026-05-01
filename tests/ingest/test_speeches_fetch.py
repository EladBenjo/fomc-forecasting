from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedtext.ingest.speeches import fetch, pipeline


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _init_speeches_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE speeches (
            id INTEGER PRIMARY KEY,
            speech_date TEXT,
            link TEXT,
            speech_text TEXT,
            processed BOOLEAN DEFAULT FALSE
        )
        """
    )
    return conn


def test_extract_text_old_uses_year_marker_without_return_to_top():
    html = """
    <html><body>
      FRB: Speech, Greenspan -- Bank supervision -- June 13, 1996
      Remarks by Chairman Alan Greenspan
      Bank supervision in a world economy
      June 13, 1996
      I am honored to present this lecture to bank supervisors from around the world.
      The banking system is becoming increasingly integrated and complex.
      Supervisors must communicate openly as markets and institutions change.
      Footnotes
      This footer should be trimmed.
    </body></html>
    """

    text = fetch._extract_text_old(_soup(html), 1996)

    assert text is not None
    assert "I am honored to present this lecture" in text
    assert "This footer should be trimmed" not in text
    assert "Return to top" not in text


def test_extract_text_old_keeps_body_when_no_end_marker_exists():
    html = """
    <html><body>
      FRB: Speech -- June 13, 1996
      Remarks by Chairman Alan Greenspan
      June 13, 1996
      This speech body has enough words to pass the useful text threshold.
      It continues with comments about supervision, financial markets, risk,
      and the responsibilities of central banks in a changing economy.
    </body></html>
    """

    text = fetch._extract_text_old(_soup(html), 1996)

    assert text is not None
    assert "This speech body has enough words" in text
    assert "changing economy" in text


def test_extract_text_new_preserves_existing_article_child_selector():
    html = """
    <html><body>
      <div id="article">
        <div>Header</div>
        <div>Metadata</div>
        <div>
          The economy has continued to expand at a moderate pace.
          Monetary policy decisions depend on the outlook and balance of risks.
          These remarks include enough content to be treated as a speech body.
          Footnotes
          This footer should be trimmed.
        </div>
      </div>
    </body></html>
    """

    text = fetch._extract_text_new(_soup(html))

    assert text is not None
    assert "Monetary policy decisions depend on the outlook" in text
    assert "This footer should be trimmed" not in text


def test_extract_text_new_prefers_article_child_before_broader_article():
    html = """
    <html><body>
      <div id="article">
        <div>Navigation text and page furniture that should not be selected.</div>
        <div>Metadata</div>
        <div>
          Governor remarks begin here with substantive comments about inflation,
          employment, financial stability, and the conduct of monetary policy.
          The selected block has enough words to pass the useful text threshold.
        </div>
        <div>
          Related links and additional page furniture add enough words to make
          the full article longer than the speech body, but this text should
          stay out of the extracted result.
        </div>
      </div>
    </body></html>
    """

    text = fetch._extract_text_new(_soup(html))

    assert text is not None
    assert "Governor remarks begin here" in text
    assert "Related links and additional page furniture" not in text


def test_extract_text_new_falls_back_to_article_body():
    html = """
    <html><body>
      <div id="article">
        <h3>Modernizing Federal Reserve Operations</h3>
        <p>PDF</p>
        <p>Share</p>
        <p>
          Thank you for having me speak today about Federal Reserve operations.
          The system has evolved over time as technology and financial markets changed.
        </p>
        <p>
          These remarks include enough substantive content to be accepted by the
          fallback extractor when the exact child selector is unavailable.
        </p>
      </div>
    </body></html>
    """

    text = fetch._extract_text_new(_soup(html))

    assert text is not None
    assert "Thank you for having me speak today" in text
    assert "\nPDF\n" not in f"\n{text}\n"
    assert "\nShare\n" not in f"\n{text}\n"


def test_fetch_run_filters_unprocessed_rows_by_year(monkeypatch):
    conn = _init_speeches_db()
    conn.executemany(
        """
        INSERT INTO speeches (id, speech_date, link, processed)
        VALUES (?, ?, ?, FALSE)
        """,
        [
            (1, "2025-12-31", "https://example.test/2025",),
            (2, "2026-01-02", "https://example.test/2026-a",),
            (3, "2026-02-03", "https://example.test/2026-b",),
        ],
    )
    conn.commit()
    seen_links: list[str] = []

    def _fake_fetch(session, link: str, year: int) -> str:
        seen_links.append(link)
        return f"speech text for {year} with enough words to satisfy the useful text threshold"

    monkeypatch.setattr(fetch, "_fetch_speech_text", _fake_fetch)
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)

    fetch.run(conn, start_year=2026, end_year=2026)

    assert seen_links == ["https://example.test/2026-a", "https://example.test/2026-b"]
    rows = conn.execute("SELECT id, processed, speech_text FROM speeches ORDER BY id").fetchall()
    assert rows[0]["processed"] == 0
    assert rows[1]["processed"] == 1
    assert rows[2]["processed"] == 1


def test_pipeline_fetch_only_passes_year_window(monkeypatch):
    captured: dict[str, object] = {}

    class _Conn:
        def close(self) -> None:
            return None

    def _fake_fetch(conn, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "get_connection", lambda: _Conn())
    monkeypatch.setattr(pipeline, "init_db", lambda conn: None)
    monkeypatch.setattr(pipeline.fetch, "run", _fake_fetch)
    monkeypatch.setattr(
        pipeline.discovery,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("discovery should not run")),
    )

    pipeline.run(fetch_only=True, start_year=2026, end_year=2026, limit=5)

    assert captured == {"limit": 5, "start_year": 2026, "end_year": 2026}
