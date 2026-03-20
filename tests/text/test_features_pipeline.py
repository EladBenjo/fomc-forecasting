"""Pipeline wiring tests for fedtext.text.features.pipeline."""

from __future__ import annotations

from pathlib import Path

from fedtext.text.features import pipeline
from fedtext.text.features.sentiment import SentimentResult


class _Conn:
    def close(self):
        return None


class _FakeClient:
    def close(self):
        return None


def _records():
    records_s = [
        {
            "doc_id": 1,
            "source_type": "speech",
            "date": "2024-01-01",
            "text": "Inflation remains elevated.",
        }
    ]
    records_d = [
        {
            "doc_id": 2,
            "source_type": "document",
            "date": "2024-01-31",
            "text": "Policy stance is restrictive.",
        }
    ]
    return records_s, records_d


def _patch_common(monkeypatch, records_s, records_d, score_fn):
    monkeypatch.setattr(pipeline, "get_connection", lambda: _Conn())
    monkeypatch.setattr(pipeline, "_load_speeches", lambda conn, limit: records_s)
    monkeypatch.setattr(pipeline, "_load_documents", lambda conn, limit: records_d)
    monkeypatch.setattr(
        pipeline.novelty_mod,
        "compute_novelty_by_type",
        lambda records: {1: 0.2, 2: 0.3},
    )
    monkeypatch.setattr(pipeline.sentiment_mod, "load_client", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(pipeline.sentiment_mod, "score_document", score_fn)


def test_run_writes_expected_schema_and_row_count(monkeypatch, tmp_path):
    records_s, records_d = _records()

    _patch_common(
        monkeypatch,
        records_s,
        records_d,
        lambda text, sents, client: SentimentResult(
            hawkish_score=0.25,
            n_hawkish=2,
            n_dovish=1,
            n_neutral=0,
            n_target_sentences=3,
        ),
    )

    monkeypatch.setattr(pipeline, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_STATE_DB", tmp_path / "features_state.sqlite3")

    captured = {}

    def _fake_atomic(df, path):
        captured["path"] = Path(path)
        captured["columns"] = list(df.columns)
        captured["rows"] = len(df)

    monkeypatch.setattr(pipeline, "_atomic_write_parquet", _fake_atomic)

    pipeline.run(source_types=["speeches", "documents"], limit=1)

    assert captured["rows"] == 2
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


def test_resume_skips_already_checkpointed_docs(monkeypatch, tmp_path):
    records_s, records_d = _records()
    call_count = {"n": 0}

    def _score(text, sents, client):
        call_count["n"] += 1
        return SentimentResult(
            hawkish_score=0.1,
            n_hawkish=1,
            n_dovish=0,
            n_neutral=0,
            n_target_sentences=1,
        )

    _patch_common(monkeypatch, records_s, records_d, _score)

    monkeypatch.setattr(pipeline, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_STATE_DB", tmp_path / "features_state.sqlite3")
    monkeypatch.setattr(pipeline, "_atomic_write_parquet", lambda df, path: None)

    pipeline.run(source_types=["speeches", "documents"], limit=1, resume=True)
    assert call_count["n"] == 2

    pipeline.run(source_types=["speeches", "documents"], limit=1, resume=True)
    assert call_count["n"] == 2


def test_reset_checkpoint_forces_recompute(monkeypatch, tmp_path):
    records_s, records_d = _records()
    call_count = {"n": 0}

    def _score(text, sents, client):
        call_count["n"] += 1
        return SentimentResult(
            hawkish_score=0.2,
            n_hawkish=1,
            n_dovish=0,
            n_neutral=0,
            n_target_sentences=1,
        )

    _patch_common(monkeypatch, records_s, records_d, _score)

    monkeypatch.setattr(pipeline, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_STATE_DB", tmp_path / "features_state.sqlite3")
    monkeypatch.setattr(pipeline, "_atomic_write_parquet", lambda df, path: None)

    pipeline.run(source_types=["speeches", "documents"], limit=1, resume=True)
    assert call_count["n"] == 2

    pipeline.run(
        source_types=["speeches", "documents"],
        limit=1,
        resume=True,
        reset_checkpoint=True,
    )
    assert call_count["n"] == 4
