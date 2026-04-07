"""Versioning and metadata tracking for Phase 4 baseline runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BaselineRunManifest:
    run_version: str
    created_at_utc: str
    git_sha: str | None
    target_column: str
    model_order: list[int]
    model_trend: str
    min_train_obs: int
    meaningful_threshold_pct: float
    model_variants_json: str
    dataset_input_path: str
    dataset_input_sha256: str | None
    dataset_input_rows: int
    splits_input_path: str
    splits_input_sha256: str | None
    output_dir: str
    predictions_output_path: str
    predictions_output_sha256: str | None
    predictions_output_rows: int
    results_output_path: str
    results_output_sha256: str | None
    paired_output_path: str
    paired_output_sha256: str | None
    summary_output_path: str
    summary_output_sha256: str | None
    config_output_path: str
    config_output_sha256: str | None
    val_start: str
    val_end: str
    test_start: str
    test_end: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_run_version(prefix: str = "phase4-t5yie-sarimax") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _init_registry_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_registry (
            run_version                 TEXT PRIMARY KEY,
            created_at_utc              TEXT NOT NULL,
            git_sha                     TEXT,
            target_column               TEXT NOT NULL,
            model_order                 TEXT NOT NULL,
            model_trend                 TEXT NOT NULL,
            min_train_obs               INTEGER NOT NULL,
            meaningful_threshold_pct    REAL NOT NULL,
            model_variants_json         TEXT NOT NULL,
            dataset_input_path          TEXT NOT NULL,
            dataset_input_sha256        TEXT,
            dataset_input_rows          INTEGER NOT NULL,
            splits_input_path           TEXT NOT NULL,
            splits_input_sha256         TEXT,
            output_dir                  TEXT NOT NULL,
            predictions_output_path     TEXT NOT NULL,
            predictions_output_sha256   TEXT,
            predictions_output_rows     INTEGER NOT NULL,
            results_output_path         TEXT NOT NULL,
            results_output_sha256       TEXT,
            paired_output_path          TEXT NOT NULL,
            paired_output_sha256        TEXT,
            summary_output_path         TEXT NOT NULL,
            summary_output_sha256       TEXT,
            config_output_path          TEXT NOT NULL,
            config_output_sha256        TEXT,
            val_start                   TEXT NOT NULL,
            val_end                     TEXT NOT NULL,
            test_start                  TEXT NOT NULL,
            test_end                    TEXT,
            metadata_json               TEXT NOT NULL
        )
        """
    )
    conn.commit()


def write_manifest(
    *,
    out_dir: Path,
    registry_path: Path,
    manifest: BaselineRunManifest,
    extra: dict[str, Any] | None = None,
) -> Path:
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    payload = manifest.to_dict()
    payload["extra"] = extra or {}

    manifest_path = manifests_dir / f"{manifest.run_version}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(registry_path)
    try:
        _init_registry_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO run_registry (
                run_version,
                created_at_utc,
                git_sha,
                target_column,
                model_order,
                model_trend,
                min_train_obs,
                meaningful_threshold_pct,
                model_variants_json,
                dataset_input_path,
                dataset_input_sha256,
                dataset_input_rows,
                splits_input_path,
                splits_input_sha256,
                output_dir,
                predictions_output_path,
                predictions_output_sha256,
                predictions_output_rows,
                results_output_path,
                results_output_sha256,
                paired_output_path,
                paired_output_sha256,
                summary_output_path,
                summary_output_sha256,
                config_output_path,
                config_output_sha256,
                val_start,
                val_end,
                test_start,
                test_end,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.run_version,
                manifest.created_at_utc,
                manifest.git_sha,
                manifest.target_column,
                json.dumps(manifest.model_order),
                manifest.model_trend,
                int(manifest.min_train_obs),
                float(manifest.meaningful_threshold_pct),
                manifest.model_variants_json,
                manifest.dataset_input_path,
                manifest.dataset_input_sha256,
                int(manifest.dataset_input_rows),
                manifest.splits_input_path,
                manifest.splits_input_sha256,
                manifest.output_dir,
                manifest.predictions_output_path,
                manifest.predictions_output_sha256,
                int(manifest.predictions_output_rows),
                manifest.results_output_path,
                manifest.results_output_sha256,
                manifest.paired_output_path,
                manifest.paired_output_sha256,
                manifest.summary_output_path,
                manifest.summary_output_sha256,
                manifest.config_output_path,
                manifest.config_output_sha256,
                manifest.val_start,
                manifest.val_end,
                manifest.test_start,
                manifest.test_end,
                json.dumps(payload["extra"]),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return manifest_path

