"""Helpers for feature dataset versioning, manifests, and comparisons."""

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
class DatasetManifest:
    dataset_version: str
    created_at_utc: str
    git_sha: str | None
    output_path: str
    output_sha256: str | None
    output_rows: int
    source_types: list[str]
    limit: int | None
    checkpoint_every: int
    resume: bool
    reset_checkpoint: bool
    cleaning_version: str
    sentence_split_version: str
    input_db_path: str
    input_db_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_dataset_version(prefix: str = "features") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
            dataset_version         TEXT PRIMARY KEY,
            created_at_utc          TEXT NOT NULL,
            git_sha                 TEXT,
            output_path             TEXT NOT NULL,
            output_sha256           TEXT,
            output_rows             INTEGER NOT NULL,
            source_types            TEXT NOT NULL,
            input_db_path           TEXT NOT NULL,
            input_db_sha256         TEXT,
            cleaning_version        TEXT NOT NULL,
            sentence_split_version  TEXT NOT NULL,
            metadata_json           TEXT NOT NULL
        )
        """
    )
    conn.commit()


def write_manifest(
    *,
    out_dir: Path,
    registry_path: Path,
    manifest: DatasetManifest,
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
                output_path,
                output_sha256,
                output_rows,
                source_types,
                input_db_path,
                input_db_sha256,
                cleaning_version,
                sentence_split_version,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.dataset_version,
                manifest.created_at_utc,
                manifest.git_sha,
                manifest.output_path,
                manifest.output_sha256,
                int(manifest.output_rows),
                json.dumps(manifest.source_types),
                manifest.input_db_path,
                manifest.input_db_sha256,
                manifest.cleaning_version,
                manifest.sentence_split_version,
                json.dumps(payload["extra"]),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return manifest_path
