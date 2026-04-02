"""Versioning and metadata tracking for model dataset build artifacts."""

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
class ModelDatasetManifest:
    dataset_version: str
    created_at_utc: str
    git_sha: str | None
    target_column: str
    monthly_feature_columns: list[str]
    features_input_path: str
    features_input_sha256: str | None
    features_input_rows: int
    target_input_path: str
    target_input_sha256: str | None
    target_input_rows: int
    output_dataset_path: str
    output_dataset_sha256: str | None
    output_dataset_rows: int
    split_output_path: str
    split_output_sha256: str | None
    summary_output_path: str | None
    summary_output_sha256: str | None
    train_end: str
    val_start: str
    val_end: str
    test_start: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_dataset_version(prefix: str = "model-dataset-t5yie") -> str:
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
        CREATE TABLE IF NOT EXISTS dataset_registry (
            dataset_version          TEXT PRIMARY KEY,
            created_at_utc           TEXT NOT NULL,
            git_sha                  TEXT,
            target_column            TEXT NOT NULL,
            monthly_feature_columns  TEXT NOT NULL,
            features_input_path      TEXT NOT NULL,
            features_input_sha256    TEXT,
            features_input_rows      INTEGER NOT NULL,
            target_input_path        TEXT NOT NULL,
            target_input_sha256      TEXT,
            target_input_rows        INTEGER NOT NULL,
            output_dataset_path      TEXT NOT NULL,
            output_dataset_sha256    TEXT,
            output_dataset_rows      INTEGER NOT NULL,
            split_output_path        TEXT NOT NULL,
            split_output_sha256      TEXT,
            summary_output_path      TEXT,
            summary_output_sha256    TEXT,
            train_end                TEXT NOT NULL,
            val_start                TEXT NOT NULL,
            val_end                  TEXT NOT NULL,
            test_start               TEXT NOT NULL,
            metadata_json            TEXT NOT NULL
        )
        """
    )
    conn.commit()


def write_manifest(
    *,
    out_dir: Path,
    registry_path: Path,
    manifest: ModelDatasetManifest,
    extra: dict[str, Any] | None = None,
) -> Path:
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    payload = manifest.to_dict()
    payload["extra"] = extra or {}

    manifest_path = manifests_dir / f"{manifest.dataset_version}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(registry_path)
    try:
        _init_registry_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO dataset_registry (
                dataset_version,
                created_at_utc,
                git_sha,
                target_column,
                monthly_feature_columns,
                features_input_path,
                features_input_sha256,
                features_input_rows,
                target_input_path,
                target_input_sha256,
                target_input_rows,
                output_dataset_path,
                output_dataset_sha256,
                output_dataset_rows,
                split_output_path,
                split_output_sha256,
                summary_output_path,
                summary_output_sha256,
                train_end,
                val_start,
                val_end,
                test_start,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.dataset_version,
                manifest.created_at_utc,
                manifest.git_sha,
                manifest.target_column,
                json.dumps(manifest.monthly_feature_columns),
                manifest.features_input_path,
                manifest.features_input_sha256,
                int(manifest.features_input_rows),
                manifest.target_input_path,
                manifest.target_input_sha256,
                int(manifest.target_input_rows),
                manifest.output_dataset_path,
                manifest.output_dataset_sha256,
                int(manifest.output_dataset_rows),
                manifest.split_output_path,
                manifest.split_output_sha256,
                manifest.summary_output_path,
                manifest.summary_output_sha256,
                manifest.train_end,
                manifest.val_start,
                manifest.val_end,
                manifest.test_start,
                json.dumps(payload["extra"]),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return manifest_path

