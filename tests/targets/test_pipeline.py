from __future__ import annotations

import json
import sqlite3

import pandas as pd

from fedtext.targets import pipeline


def test_apply_transform_diff1_returns_expected_values():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [2.0, 2.5, 1.5],
        }
    )

    transformed = pipeline.apply_transform(df, transform_id="diff1")

    assert transformed["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert transformed["value_transformed"].tolist() == [0.5, -1.0]


def test_run_writes_parquets_manifest_and_registry(monkeypatch, tmp_path):
    fetched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [2.0, 2.2, 2.8],
        }
    )

    monkeypatch.setattr(pipeline, "TARGETS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "fetch_series", lambda series_id, start, end: fetched)
    monkeypatch.setattr(pipeline, "get_git_sha", lambda repo_root: "abc123")

    pipeline.run(
        series_id="T5YIE",
        transform_id="diff1",
        dataset_version="targets-t5yie-test",
        start="2024-01-01",
        end="2024-01-03",
        write_manifest_files=True,
    )

    raw_path = tmp_path / "t5yie_raw.parquet"
    transformed_path = tmp_path / "t5yie_diff1.parquet"
    manifest_path = tmp_path / "manifests" / "targets-t5yie-test.json"
    registry_path = tmp_path / "dataset_registry.sqlite3"

    assert raw_path.exists()
    assert transformed_path.exists()
    assert manifest_path.exists()
    assert registry_path.exists()

    raw_df = pd.read_parquet(raw_path)
    transformed_df = pd.read_parquet(transformed_path)
    assert list(raw_df.columns) == ["date", "t5yie"]
    assert list(transformed_df.columns) == ["date", "t5yie_diff1"]
    assert len(raw_df) == 3
    assert len(transformed_df) == 2

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["series_id"] == "T5YIE"
    assert payload["transform_id"] == "diff1"
    assert payload["raw_output_rows"] == 3
    assert payload["transformed_output_rows"] == 2

    conn = sqlite3.connect(registry_path)
    try:
        row = conn.execute(
            """
            SELECT dataset_version, series_id, transform_id, raw_output_rows, transformed_output_rows
            FROM dataset_registry
            WHERE dataset_version = 'targets-t5yie-test'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == ("targets-t5yie-test", "T5YIE", "diff1", 3, 2)


def test_run_rerun_creates_new_dataset_versions_when_auto(monkeypatch, tmp_path):
    fetched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [1.0, 1.5, 1.8],
        }
    )
    versions = iter(["targets-v1", "targets-v2"])

    monkeypatch.setattr(pipeline, "TARGETS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "fetch_series", lambda series_id, start, end: fetched)
    monkeypatch.setattr(pipeline, "get_git_sha", lambda repo_root: "abc123")
    monkeypatch.setattr(
        pipeline, "default_dataset_version", lambda prefix="targets": next(versions)
    )

    pipeline.run(series_id="T5YIE", transform_id="diff1", write_manifest_files=True)
    pipeline.run(series_id="T5YIE", transform_id="diff1", write_manifest_files=True)

    conn = sqlite3.connect(tmp_path / "dataset_registry.sqlite3")
    try:
        count = conn.execute("SELECT COUNT(*) FROM dataset_registry").fetchone()[0]
        rows = conn.execute(
            "SELECT dataset_version FROM dataset_registry ORDER BY dataset_version"
        ).fetchall()
    finally:
        conn.close()

    assert count == 2
    assert rows == [("targets-v1",), ("targets-v2",)]

