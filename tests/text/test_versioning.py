from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fedtext.text.features.versioning import DatasetManifest, write_manifest


def test_write_manifest_creates_json_and_registry(tmp_path: Path):
    out_dir = tmp_path / "doc_level"
    registry = out_dir / "dataset_registry.sqlite3"

    output_file = out_dir / "features.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"dummy")

    manifest = DatasetManifest(
        dataset_version="features-20260321T000000Z",
        created_at_utc="2026-03-21T00:00:00+00:00",
        git_sha="abc123",
        output_path=str(output_file),
        output_sha256="deadbeef",
        output_rows=42,
        source_types=["speeches", "documents"],
        limit=None,
        checkpoint_every=25,
        resume=True,
        reset_checkpoint=False,
        cleaning_version="1.1.0",
        sentence_split_version="2.0.0",
        input_db_path="/tmp/fedtext.db",
        input_db_sha256="beadfeed",
    )

    manifest_path = write_manifest(
        out_dir=out_dir,
        registry_path=registry,
        manifest=manifest,
        extra={"max_retries": 5},
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "features-20260321T000000Z"
    assert payload["extra"]["max_retries"] == 5

    conn = sqlite3.connect(registry)
    try:
        row = conn.execute(
            "SELECT dataset_version, cleaning_version, sentence_split_version, output_rows FROM dataset_registry"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("features-20260321T000000Z", "1.1.0", "2.0.0", 42)
