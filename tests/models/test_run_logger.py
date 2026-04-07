from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from models.tracking.run_logger import BaselineRunManifest, write_manifest


def _manifest(run_version: str, out_dir: Path, *, prediction_rows: int) -> BaselineRunManifest:
    return BaselineRunManifest(
        run_version=run_version,
        created_at_utc="2026-04-07T00:00:00+00:00",
        git_sha="abc123",
        target_column="t5yie_diff1",
        model_order=[1, 0, 0],
        model_trend="c",
        min_train_obs=36,
        meaningful_threshold_pct=2.0,
        model_variants_json='{"baseline_univariate": []}',
        dataset_input_path=str(out_dir / "model_dataset_t5yie.parquet"),
        dataset_input_sha256="dataset-hash",
        dataset_input_rows=279,
        splits_input_path=str(out_dir / "time_splits.json"),
        splits_input_sha256="splits-hash",
        output_dir=str(out_dir / run_version),
        predictions_output_path=str(out_dir / run_version / "predictions.parquet"),
        predictions_output_sha256="pred-hash",
        predictions_output_rows=prediction_rows,
        results_output_path=str(out_dir / run_version / "results_table.json"),
        results_output_sha256="results-hash",
        paired_output_path=str(out_dir / run_version / "paired_comparison.json"),
        paired_output_sha256="paired-hash",
        summary_output_path=str(out_dir / run_version / "run_summary.json"),
        summary_output_sha256="summary-hash",
        config_output_path=str(out_dir / run_version / "run_config.json"),
        config_output_sha256="config-hash",
        val_start="2017-01-01",
        val_end="2020-12-31",
        test_start="2021-01-01",
        test_end=None,
    )


def test_write_manifest_creates_json_and_registry(tmp_path: Path):
    out_dir = tmp_path / "models"
    registry_path = out_dir / "run_registry.sqlite3"

    manifest_path = write_manifest(
        out_dir=out_dir,
        registry_path=registry_path,
        manifest=_manifest("phase4-run-1", out_dir, prediction_rows=111),
        extra={"best_exog_variant": "exog_share_variant"},
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_version"] == "phase4-run-1"
    assert payload["predictions_output_rows"] == 111
    assert payload["extra"]["best_exog_variant"] == "exog_share_variant"

    conn = sqlite3.connect(registry_path)
    try:
        row = conn.execute(
            """
            SELECT run_version, target_column, predictions_output_rows
            FROM run_registry
            WHERE run_version = 'phase4-run-1'
            """
        ).fetchone()
    finally:
        conn.close()

    assert row == ("phase4-run-1", "t5yie_diff1", 111)


def test_write_manifest_upserts_run_version(tmp_path: Path):
    out_dir = tmp_path / "models"
    registry_path = out_dir / "run_registry.sqlite3"

    write_manifest(
        out_dir=out_dir,
        registry_path=registry_path,
        manifest=_manifest("phase4-run-same", out_dir, prediction_rows=100),
    )
    write_manifest(
        out_dir=out_dir,
        registry_path=registry_path,
        manifest=_manifest("phase4-run-same", out_dir, prediction_rows=101),
    )

    conn = sqlite3.connect(registry_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM run_registry").fetchone()[0]
        row = conn.execute(
            """
            SELECT predictions_output_rows
            FROM run_registry
            WHERE run_version = 'phase4-run-same'
            """
        ).fetchone()
    finally:
        conn.close()

    assert count == 1
    assert row == (101,)

