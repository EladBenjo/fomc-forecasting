from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from fedtext.targets import store


def _write_features(path: Path) -> None:
    df = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "source_type": ["speech", "document"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "hawkish_score": [0.2, -0.1],
        }
    )
    df.to_parquet(path, index=False)


def _write_targets(targets_dir: Path, *, series_id: str = "T5YIE", transform_id: str = "diff1") -> None:
    slug = series_id.lower()
    transform_slug = transform_id.lower()
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            slug: [2.0, 2.1, 2.2],
        }
    )
    transformed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            f"{slug}_{transform_slug}": [0.1, 0.1],
        }
    )
    targets_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(targets_dir / f"{slug}_raw.parquet", index=False)
    transformed.to_parquet(targets_dir / f"{slug}_{transform_slug}.parquet", index=False)


def test_build_eda_db_writes_expected_tables(tmp_path: Path):
    features_path = tmp_path / "features.parquet"
    targets_dir = tmp_path / "targets"
    db_path = tmp_path / "eda.sqlite3"
    _write_features(features_path)
    _write_targets(targets_dir)

    out = store.build_eda_db(
        db_path=db_path,
        features_parquet=features_path,
        series_id="T5YIE",
        transform_id="diff1",
        auto_fetch_missing=False,
        targets_dir=targets_dir,
    )

    assert out == db_path
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "features_doc_level" in tables
        assert "target_raw" in tables
        assert "target_transformed" in tables
        assert "build_metadata" in tables

        features_rows = conn.execute("SELECT COUNT(*) FROM features_doc_level").fetchone()[0]
        raw_rows = conn.execute("SELECT COUNT(*) FROM target_raw").fetchone()[0]
        transformed_rows = conn.execute("SELECT COUNT(*) FROM target_transformed").fetchone()[0]
        metadata_rows = conn.execute("SELECT COUNT(*) FROM build_metadata").fetchone()[0]
    finally:
        conn.close()

    assert features_rows == 2
    assert raw_rows == 3
    assert transformed_rows == 2
    assert metadata_rows == 1


def test_build_eda_db_auto_fetches_missing_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    features_path = tmp_path / "features.parquet"
    targets_dir = tmp_path / "targets"
    db_path = tmp_path / "eda.sqlite3"
    _write_features(features_path)

    called = {"n": 0}

    def _fake_run(*, series_id, transform_id, start, end, write_manifest_files):
        del start, end, write_manifest_files
        called["n"] += 1
        _write_targets(targets_dir, series_id=series_id, transform_id=transform_id)

    monkeypatch.setattr(store.targets_pipeline, "run", _fake_run)

    store.build_eda_db(
        db_path=db_path,
        features_parquet=features_path,
        series_id="T5YIE",
        transform_id="diff1",
        auto_fetch_missing=True,
        targets_dir=targets_dir,
    )

    assert called["n"] == 1
    assert db_path.exists()


def test_build_eda_db_fails_fast_when_targets_missing_and_auto_fetch_disabled(tmp_path: Path):
    features_path = tmp_path / "features.parquet"
    targets_dir = tmp_path / "targets"
    db_path = tmp_path / "eda.sqlite3"
    _write_features(features_path)

    with pytest.raises(FileNotFoundError, match="python -m fedtext.targets.pipeline"):
        store.build_eda_db(
            db_path=db_path,
            features_parquet=features_path,
            series_id="T5YIE",
            transform_id="diff1",
            auto_fetch_missing=False,
            targets_dir=targets_dir,
        )

