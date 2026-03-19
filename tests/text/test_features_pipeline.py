"""Pipeline wiring tests for fedtext.text.features.pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedtext.text.features import pipeline
from fedtext.text.features.sentiment import SentimentResult


def test_run_writes_expected_schema_and_row_count(monkeypatch, tmp_path):
    records_s = [
        {
            "doc_id": "s1",
            "source_type": "speech",
            "date": "2024-01-01",
            "text": "Inflation remains elevated.",
        }
    ]
    records_d = [
        {
            "doc_id": "d1",
            "source_type": "document",
            "date": "2024-01-31",
            "text": "Policy stance is restrictive.",
        }
    ]

    class _Conn:
        def close(self):
            return None

    monkeypatch.setattr(pipeline, "get_connection", lambda: _Conn())
    monkeypatch.setattr(pipeline, "_load_speeches", lambda conn, limit: records_s)
    monkeypatch.setattr(pipeline, "_load_documents", lambda conn, limit: records_d)
    monkeypatch.setattr(
        pipeline.novelty_mod,
        "compute_novelty_by_type",
        lambda records: {"s1": 0.2, "d1": 0.3},
    )
    monkeypatch.setattr(pipeline.sentiment_mod, "load_client", lambda: object())
    monkeypatch.setattr(
        pipeline.sentiment_mod,
        "score_document",
        lambda text, sents, client: SentimentResult(
            hawkish_score=0.25,
            n_hawkish=2,
            n_dovish=1,
            n_neutral=0,
            n_target_sentences=3,
        ),
    )

    monkeypatch.setattr(pipeline, "_OUT_DIR", tmp_path)

    captured = {}

    def _fake_to_parquet(self, path, index=False):
        captured["path"] = Path(path)
        captured["index"] = index
        captured["columns"] = list(self.columns)
        captured["rows"] = len(self)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=True)

    pipeline.run(source_types=["speeches", "documents"], limit=1)

    assert captured["rows"] == 2
    assert captured["index"] is False
    assert captured["path"] == tmp_path / "features.parquet"
    assert captured["columns"] == [
        "doc_id",
        "source_type",
        "date",
        "hawkish_score",
        "n_hawkish",
        "n_dovish",
        "n_neutral",
        "n_target_sentences",
        "novelty",
    ]
