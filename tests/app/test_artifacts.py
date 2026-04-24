from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from app.lib import artifacts


def _write_features(repo_root: Path) -> Path:
    path = repo_root / "data" / "features" / "doc_level" / "features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "source_type": ["speech", "document"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "hawkish_score": [0.2, -0.1],
            "novelty": [0.3, 0.4],
        }
    )
    df.to_parquet(path, index=False)
    return path


def _write_backfilled_features(repo_root: Path) -> Path:
    path = repo_root / "data" / "features" / "doc_level" / "features_backfilled_with_doc_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "doc_id": [10],
            "source_type": ["speech"],
            "date": pd.to_datetime(["2024-02-01"]),
            "hawkish_score": [0.75],
            "novelty": [0.8],
            "text_length_words": [1200],
            "role": ["Chairman"],
            "target_sentences_ratio": [0.4],
        }
    )
    df.to_parquet(path, index=False)
    return path


def _write_fedtext_catalog(repo_root: Path) -> Path:
    db_path = repo_root / "data" / "catalog" / "fedtext.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE speeches (
                id INTEGER PRIMARY KEY,
                speech_date TEXT,
                title TEXT,
                speaker TEXT,
                event TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                category TEXT,
                meeting_date TEXT,
                meeting_label TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO speeches (id, speech_date, title, speaker, event)
            VALUES (1, '2024-01-01', 'Speech Title', 'Chair X', 'Event A')
            """
        )
        conn.execute(
            """
            INSERT INTO documents (id, doc_id, category, meeting_date, meeting_label)
            VALUES (2, 'D1', 'minutes', '2024-01-01', 'FOMC Minutes')
            """
        )
        conn.commit()
    return db_path


def _write_phase4_run(repo_root: Path, run_version: str, *, complete: bool = True) -> Path:
    run_dir = repo_root / "data" / "models" / "baselines" / "t5yie" / run_version
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "run_version": [run_version, run_version],
            "model_variant": ["baseline_univariate", "baseline_univariate"],
            "split": ["test", "test"],
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "actual": [0.1, -0.2],
            "pred": [0.05, -0.1],
        }
    )
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "results_table.json").write_text(
        json.dumps([{"split": "test", "model_variant": "baseline_univariate", "rmse": 0.1}]),
        encoding="utf-8",
    )
    (run_dir / "paired_comparison.json").write_text(
        json.dumps([{"split": "test", "exog_variant": "exog_share_variant", "n_common_dates": 2}]),
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps({"best_exog_variant": "exog_share_variant", "is_meaningful": False}),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"model_order": [1, 0, 0], "model_variants": {"baseline_univariate": []}}),
        encoding="utf-8",
    )

    if not complete:
        (run_dir / "run_summary.json").unlink()

    return run_dir


def test_list_phase4_runs_and_latest_selection(tmp_path: Path):
    _write_phase4_run(tmp_path, "phase4-20260401T000000Z", complete=True)
    _write_phase4_run(tmp_path, "phase4-20260402T000000Z", complete=True)
    _write_phase4_run(tmp_path, "phase4-20260403T000000Z", complete=False)

    assert artifacts.list_phase4_runs(repo_root=tmp_path, require_complete=False) == [
        "phase4-20260401T000000Z",
        "phase4-20260402T000000Z",
        "phase4-20260403T000000Z",
    ]
    assert artifacts.list_phase4_runs(repo_root=tmp_path, require_complete=True) == [
        "phase4-20260401T000000Z",
        "phase4-20260402T000000Z",
    ]
    assert artifacts.latest_phase4_run(repo_root=tmp_path, require_complete=True) == "phase4-20260402T000000Z"


def test_load_phase4_run_artifacts_with_fixture_run(tmp_path: Path):
    _write_phase4_run(tmp_path, "phase4-20260405T000000Z", complete=True)

    payload = artifacts.load_phase4_run_artifacts(repo_root=tmp_path)
    assert payload["run_version"] == "phase4-20260405T000000Z"
    assert not payload["predictions"].empty
    assert set(payload["predictions"].columns) == {
        "run_version",
        "model_variant",
        "split",
        "date",
        "actual",
        "pred",
    }
    assert "rmse" in payload["results_table"].columns


def test_load_phase4_run_artifacts_raises_on_incomplete_run(tmp_path: Path):
    _write_phase4_run(tmp_path, "phase4-20260405T000000Z", complete=False)

    with pytest.raises(FileNotFoundError, match="No complete Phase 4 runs found"):
        artifacts.load_phase4_run_artifacts(repo_root=tmp_path)


def test_build_status_snapshot_missing_artifacts_has_guidance(tmp_path: Path):
    snapshot = artifacts.build_status_snapshot(repo_root=tmp_path)

    assert snapshot["ready_count"] == 0
    assert snapshot["runs"]["latest_complete"] is None
    command_map = {check["key"]: check["remediation_command"] for check in snapshot["checks"]}
    assert command_map["features"] == "python -m fedtext.text.features.pipeline"
    assert command_map["phase4"] == "python -m models.baselines.sarimax"


def test_load_features_dataframe(tmp_path: Path):
    _write_features(tmp_path)
    out = artifacts.load_features_dataframe(repo_root=tmp_path)
    assert len(out) == 2
    assert out["date"].is_monotonic_increasing


def test_load_features_dataframe_prefers_backfilled_when_available(tmp_path: Path):
    _write_features(tmp_path)
    _write_backfilled_features(tmp_path)

    out = artifacts.load_features_dataframe(repo_root=tmp_path, prefer_backfilled=True)
    assert len(out) == 1
    assert out.attrs["source_path"].endswith("features_backfilled_with_doc_features.parquet")
    assert "text_length_words" in out.columns

    fallback = artifacts.load_features_dataframe(repo_root=tmp_path, prefer_backfilled=False)
    assert len(fallback) == 2
    assert fallback.attrs["source_path"].endswith("features.parquet")


def test_load_optional_document_metadata_from_fedtext_db(tmp_path: Path):
    _write_fedtext_catalog(tmp_path)

    metadata = artifacts.load_optional_document_metadata(repo_root=tmp_path)
    assert not metadata.empty
    assert set(metadata["source_type"]) == {"speech", "document"}
    assert {"title", "speaker", "event"}.issubset(metadata.columns)


def test_get_artifact_freshness_includes_expected_rows(tmp_path: Path):
    _write_features(tmp_path)
    freshness = artifacts.get_artifact_freshness(repo_root=tmp_path)
    assert {"artifact", "path", "exists", "updated_at_utc"}.issubset(freshness.columns)
    assert "doc_features" in set(freshness["artifact"])

