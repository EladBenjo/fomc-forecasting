from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fedtext.targets.versioning import TargetDatasetManifest, write_manifest


def test_write_manifest_creates_json_and_registry(tmp_path: Path):
    out_dir = tmp_path / "targets"
    registry = out_dir / "dataset_registry.sqlite3"

    raw_file = out_dir / "t5yie_raw.parquet"
    transformed_file = out_dir / "t5yie_diff1.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file.write_bytes(b"raw")
    transformed_file.write_bytes(b"transformed")

    manifest = TargetDatasetManifest(
        dataset_version="targets-t5yie-20260323T000000Z",
        created_at_utc="2026-03-23T00:00:00+00:00",
        git_sha="abc123",
        series_id="T5YIE",
        transform_id="diff1",
        raw_output_path=str(raw_file),
        raw_output_sha256="aaa",
        raw_output_rows=123,
        transformed_output_path=str(transformed_file),
        transformed_output_sha256="bbb",
        transformed_output_rows=122,
        fetch_start="2010-01-01",
        fetch_end="2025-12-31",
    )

    manifest_path = write_manifest(
        out_dir=out_dir,
        registry_path=registry,
        manifest=manifest,
        extra={"fred_api_key_env": "FRED_API_KEY"},
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "targets-t5yie-20260323T000000Z"
    assert payload["series_id"] == "T5YIE"
    assert payload["transform_id"] == "diff1"
    assert payload["extra"]["fred_api_key_env"] == "FRED_API_KEY"

    conn = sqlite3.connect(registry)
    try:
        row = conn.execute(
            """
            SELECT dataset_version, series_id, transform_id, raw_output_rows, transformed_output_rows
            FROM dataset_registry
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == ("targets-t5yie-20260323T000000Z", "T5YIE", "diff1", 123, 122)


def test_write_manifest_upserts_existing_dataset_version(tmp_path: Path):
    out_dir = tmp_path / "targets"
    registry = out_dir / "dataset_registry.sqlite3"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_v1 = TargetDatasetManifest(
        dataset_version="targets-same-version",
        created_at_utc="2026-03-23T00:00:00+00:00",
        git_sha="sha1",
        series_id="T5YIE",
        transform_id="diff1",
        raw_output_path=str(out_dir / "t5yie_raw.parquet"),
        raw_output_sha256="raw1",
        raw_output_rows=100,
        transformed_output_path=str(out_dir / "t5yie_diff1.parquet"),
        transformed_output_sha256="tr1",
        transformed_output_rows=99,
        fetch_start=None,
        fetch_end=None,
    )
    write_manifest(out_dir=out_dir, registry_path=registry, manifest=manifest_v1)

    manifest_v2 = TargetDatasetManifest(
        dataset_version="targets-same-version",
        created_at_utc="2026-03-23T00:05:00+00:00",
        git_sha="sha2",
        series_id="T5YIE",
        transform_id="diff1",
        raw_output_path=str(out_dir / "t5yie_raw.parquet"),
        raw_output_sha256="raw2",
        raw_output_rows=101,
        transformed_output_path=str(out_dir / "t5yie_diff1.parquet"),
        transformed_output_sha256="tr2",
        transformed_output_rows=100,
        fetch_start="2015-01-01",
        fetch_end="2020-12-31",
    )
    write_manifest(out_dir=out_dir, registry_path=registry, manifest=manifest_v2)

    conn = sqlite3.connect(registry)
    try:
        count = conn.execute("SELECT COUNT(*) FROM dataset_registry").fetchone()[0]
        row = conn.execute(
            """
            SELECT git_sha, raw_output_rows, transformed_output_rows, fetch_start, fetch_end
            FROM dataset_registry
            WHERE dataset_version = 'targets-same-version'
            """
        ).fetchone()
    finally:
        conn.close()

    assert count == 1
    assert row == ("sha2", 101, 100, "2015-01-01", "2020-12-31")

