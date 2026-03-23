"""Helpers for target-series dataset versioning and metadata tracking."""

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
class TargetDatasetManifest:
    dataset_version: str
    created_at_utc: str
    git_sha: str | None
    series_id: str
    transform_id: str
    raw_output_path: str
    raw_output_sha256: str | None
    raw_output_rows: int
    transformed_output_path: str
    transformed_output_sha256: str | None
    transformed_output_rows: int
    fetch_start: str | None
    fetch_end: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_dataset_version(prefix: str = "targets") -> str:
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
            dataset_version           TEXT PRIMARY KEY,
            created_at_utc            TEXT NOT NULL,
            git_sha                   TEXT,
            series_id                 TEXT NOT NULL,
            transform_id              TEXT NOT NULL,
            raw_output_path           TEXT NOT NULL,
            raw_output_sha256         TEXT,
            raw_output_rows           INTEGER NOT NULL,
            transformed_output_path   TEXT NOT NULL,
            transformed_output_sha256 TEXT,
            transformed_output_rows   INTEGER NOT NULL,
            fetch_start               TEXT,
            fetch_end                 TEXT,
            metadata_json             TEXT NOT NULL
        )
        """
    )
    conn.commit()


def write_manifest(
    *,
    out_dir: Path,
    registry_path: Path,
    manifest: TargetDatasetManifest,
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
                series_id,
                transform_id,
                raw_output_path,
                raw_output_sha256,
                raw_output_rows,
                transformed_output_path,
                transformed_output_sha256,
                transformed_output_rows,
                fetch_start,
                fetch_end,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.dataset_version,
                manifest.created_at_utc,
                manifest.git_sha,
                manifest.series_id,
                manifest.transform_id,
                manifest.raw_output_path,
                manifest.raw_output_sha256,
                int(manifest.raw_output_rows),
                manifest.transformed_output_path,
                manifest.transformed_output_sha256,
                int(manifest.transformed_output_rows),
                manifest.fetch_start,
                manifest.fetch_end,
                json.dumps(payload["extra"]),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return manifest_path

