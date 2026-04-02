from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from datasets.build_dataset.versioning import ModelDatasetManifest, write_manifest


def _manifest(dataset_version: str, out_dir: Path, *, rows: int) -> ModelDatasetManifest:
    return ModelDatasetManifest(
        dataset_version=dataset_version,
        created_at_utc="2026-04-02T00:00:00+00:00",
        git_sha="abc123",
        target_column="t5yie_diff1",
        monthly_feature_columns=["hawkish_score", "novelty", "doc_count"],
        features_input_path=str(out_dir / "features.parquet"),
        features_input_sha256="fhash",
        features_input_rows=1695,
        target_input_path=str(out_dir / "t5yie_diff1.parquet"),
        target_input_sha256="thash",
        target_input_rows=5808,
        output_dataset_path=str(out_dir / "model_dataset_t5yie.parquet"),
        output_dataset_sha256="ohash",
        output_dataset_rows=rows,
        split_output_path=str(out_dir / "time_splits.json"),
        split_output_sha256="shash",
        summary_output_path=str(out_dir / "summary.json"),
        summary_output_sha256="sumhash",
        train_end="2016-12-31",
        val_start="2017-01-01",
        val_end="2020-12-31",
        test_start="2021-01-01",
    )


def test_write_manifest_creates_json_and_registry(tmp_path: Path):
    out_dir = tmp_path / "targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "model_dataset_registry.sqlite3"

    manifest = _manifest("model-dataset-t5yie-20260402T000000Z", out_dir, rows=279)
    manifest_path = write_manifest(
        out_dir=out_dir,
        registry_path=registry_path,
        manifest=manifest,
        extra={"split_counts": {"train": 168, "val": 48, "test": 63}},
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "model-dataset-t5yie-20260402T000000Z"
    assert payload["output_dataset_rows"] == 279
    assert payload["extra"]["split_counts"]["test"] == 63

    conn = sqlite3.connect(registry_path)
    try:
        row = conn.execute(
            """
            SELECT dataset_version, target_column, output_dataset_rows
            FROM dataset_registry
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == ("model-dataset-t5yie-20260402T000000Z", "t5yie_diff1", 279)


def test_write_manifest_upserts_dataset_version(tmp_path: Path):
    out_dir = tmp_path / "targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "model_dataset_registry.sqlite3"

    write_manifest(
        out_dir=out_dir,
        registry_path=registry_path,
        manifest=_manifest("model-dataset-same", out_dir, rows=100),
    )
    write_manifest(
        out_dir=out_dir,
        registry_path=registry_path,
        manifest=_manifest("model-dataset-same", out_dir, rows=101),
    )

    conn = sqlite3.connect(registry_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM dataset_registry").fetchone()[0]
        row = conn.execute(
            """
            SELECT output_dataset_rows
            FROM dataset_registry
            WHERE dataset_version = 'model-dataset-same'
            """
        ).fetchone()
    finally:
        conn.close()

    assert count == 1
    assert row == (101,)

