"""Read-only artifact access layer for the Streamlit economist dashboard."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH_REL = Path("data/features/doc_level/features.parquet")
FEATURES_BACKFILLED_PATH_REL = Path("data/features/doc_level/features_backfilled_with_doc_features.parquet")
MODEL_DATASET_PATH_REL = Path("data/targets/model_dataset_t5yie.parquet")
SPLITS_PATH_REL = Path("data/splits/time_splits.json")
PHASE4_RUNS_ROOT_REL = Path("data/models/baselines/t5yie")
XGBOOST_RUNS_ROOT_REL = Path("data/models/xgboost/t5yie")
FEDTEXT_CATALOG_DB_REL = Path("data/catalog/fedtext.db")
SPEECHES_CATALOG_DB_REL = Path("data/catalog/speeches.db")
DOCUMENTS_CATALOG_DB_REL = Path("data/catalog/catalog.sqlite")

PHASE4_REQUIRED_FILES: tuple[str, ...] = (
    "predictions.parquet",
    "results_table.json",
    "paired_comparison.json",
    "run_summary.json",
    "run_config.json",
)
XGBOOST_REQUIRED_FILES: tuple[str, ...] = (
    "predictions.parquet",
    "results_table.json",
    "run_summary.json",
    "run_config.json",
    "feature_importance.json",
    "feature_schema.json",
    "model.json",
)


def remediation_commands() -> dict[str, str]:
    """Canonical CLI guidance for missing artifacts."""
    return {
        "features": "python -m fedtext.text.features.pipeline",
        "model_dataset": "python -m datasets.build_dataset.builder",
        "phase4": "python -m models.baselines.sarimax",
        "xgboost": "python -m models.ml.xgboost",
        "app": "streamlit run app/main.py",
    }


def _root(repo_root: Path | None = None) -> Path:
    return Path(repo_root) if repo_root is not None else REPO_ROOT


def _paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = _root(repo_root)
    return {
        "repo_root": root,
        "features": root / FEATURES_PATH_REL,
        "features_backfilled": root / FEATURES_BACKFILLED_PATH_REL,
        "model_dataset": root / MODEL_DATASET_PATH_REL,
        "splits": root / SPLITS_PATH_REL,
        "phase4_runs_root": root / PHASE4_RUNS_ROOT_REL,
        "xgboost_runs_root": root / XGBOOST_RUNS_ROOT_REL,
        "fedtext_catalog_db": root / FEDTEXT_CATALOG_DB_REL,
        "speeches_catalog_db": root / SPEECHES_CATALOG_DB_REL,
        "documents_catalog_db": root / DOCUMENTS_CATALOG_DB_REL,
    }


def _feature_candidates(paths: dict[str, Path], *, prefer_backfilled: bool) -> list[Path]:
    primary = paths["features_backfilled"] if prefer_backfilled else paths["features"]
    secondary = paths["features"] if prefer_backfilled else paths["features_backfilled"]
    return [primary, secondary]


def resolve_features_path(
    *,
    repo_root: Path | None = None,
    prefer_backfilled: bool = True,
) -> Path:
    """Resolve doc-level feature file with backfilled-first fallback policy."""
    paths = _paths(repo_root)
    for candidate in _feature_candidates(paths, prefer_backfilled=prefer_backfilled):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing feature artifact. Checked: "
        f"{paths['features_backfilled']} and {paths['features']}"
    )


def _normalize_doc_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64")


def _normalize_date_column(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=[date_col]).sort_values(date_col, kind="mergesort").reset_index(drop=True)
    return out


def list_phase4_runs(
    *,
    repo_root: Path | None = None,
    require_complete: bool = False,
) -> list[str]:
    """List phase4 run versions under data/models/baselines/t5yie."""
    return _list_model_runs(
        runs_root=_paths(repo_root)["phase4_runs_root"],
        required_files=PHASE4_REQUIRED_FILES,
        require_complete=require_complete,
    )


def list_xgboost_runs(
    *,
    repo_root: Path | None = None,
    require_complete: bool = False,
) -> list[str]:
    """List XGBoost run versions under data/models/xgboost/t5yie."""
    return _list_model_runs(
        runs_root=_paths(repo_root)["xgboost_runs_root"],
        required_files=XGBOOST_REQUIRED_FILES,
        require_complete=require_complete,
    )


def _list_model_runs(
    *,
    runs_root: Path,
    required_files: tuple[str, ...],
    require_complete: bool,
) -> list[str]:
    if not runs_root.exists() or not runs_root.is_dir():
        return []

    discovered: list[str] = []
    for candidate in sorted(runs_root.iterdir(), key=lambda p: p.name):
        if not candidate.is_dir():
            continue
        missing = _missing_files(candidate, required_files)
        if require_complete and missing:
            continue
        discovered.append(candidate.name)
    return discovered


def latest_phase4_run(
    *,
    repo_root: Path | None = None,
    require_complete: bool = True,
) -> str | None:
    """Deterministically choose the latest run by lexicographic run_version."""
    runs = list_phase4_runs(repo_root=repo_root, require_complete=require_complete)
    return runs[-1] if runs else None


def latest_xgboost_run(
    *,
    repo_root: Path | None = None,
    require_complete: bool = True,
) -> str | None:
    """Deterministically choose the latest XGBoost run by lexicographic run_version."""
    runs = list_xgboost_runs(repo_root=repo_root, require_complete=require_complete)
    return runs[-1] if runs else None


def missing_phase4_files(run_dir: Path) -> list[str]:
    """Return missing required files for a run directory."""
    return _missing_files(run_dir, PHASE4_REQUIRED_FILES)


def missing_xgboost_files(run_dir: Path) -> list[str]:
    """Return missing required files for an XGBoost run directory."""
    return _missing_files(run_dir, XGBOOST_REQUIRED_FILES)


def _missing_files(run_dir: Path, required_files: tuple[str, ...]) -> list[str]:
    return [name for name in required_files if not (run_dir / name).exists()]


def load_features_dataframe(
    *,
    repo_root: Path | None = None,
    prefer_backfilled: bool = True,
) -> pd.DataFrame:
    """Load the doc-level feature artifact, preferring backfilled schema when available."""
    features_path = resolve_features_path(repo_root=repo_root, prefer_backfilled=prefer_backfilled)
    df = pd.read_parquet(features_path)
    if "source_type" in df.columns:
        df["source_type"] = df["source_type"].astype(str).str.strip().str.lower()
    if "doc_id" in df.columns:
        df["doc_id"] = _normalize_doc_id(df["doc_id"])
    df = _normalize_date_column(df, date_col="date")
    df.attrs["source_path"] = str(features_path)
    return df


def load_model_dataset_dataframe(*, repo_root: Path | None = None) -> pd.DataFrame:
    """Load the monthly model dataset used by benchmark/model dashboards."""
    dataset_path = _paths(repo_root)["model_dataset"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset artifact: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    return _normalize_date_column(df, date_col="date")


def load_time_splits_payload(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Load time split metadata from JSON."""
    splits_path = _paths(repo_root)["splits"]
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing time split artifact: {splits_path}")
    return json.loads(splits_path.read_text(encoding="utf-8"))


def _read_sqlite_query(db_path: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def _empty_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["source_type", "doc_id", "title", "speaker", "event"])


def _safe_table_exists(db_path: Path, table: str) -> bool:
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
    """
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(query, (table,)).fetchone()
            return row is not None
    except sqlite3.Error:
        return False


def _normalize_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_metadata_frame()
    out = df.copy()
    for col in ("title", "speaker", "event"):
        if col not in out.columns:
            out[col] = None
    out["source_type"] = out["source_type"].astype(str).str.strip().str.lower()
    out["doc_id"] = _normalize_doc_id(out["doc_id"])
    out = out.dropna(subset=["doc_id"])
    out = out.drop_duplicates(subset=["source_type", "doc_id"], keep="first")
    return out[["source_type", "doc_id", "title", "speaker", "event"]].reset_index(drop=True)


def load_optional_document_metadata(*, repo_root: Path | None = None) -> pd.DataFrame:
    """Load optional title/speaker/event metadata for event tables."""
    paths = _paths(repo_root)
    frames: list[pd.DataFrame] = []

    fedtext_db = paths["fedtext_catalog_db"]
    if fedtext_db.exists():
        try:
            if _safe_table_exists(fedtext_db, "speeches"):
                frames.append(
                    _read_sqlite_query(
                        fedtext_db,
                        """
                        SELECT
                            'speech' AS source_type,
                            id AS doc_id,
                            title,
                            speaker,
                            event
                        FROM speeches
                        """,
                    )
                )
            if _safe_table_exists(fedtext_db, "documents"):
                frames.append(
                    _read_sqlite_query(
                        fedtext_db,
                        """
                        SELECT
                            'document' AS source_type,
                            id AS doc_id,
                            COALESCE(meeting_label, category, doc_id) AS title,
                            NULL AS speaker,
                            category AS event
                        FROM documents
                        """,
                    )
                )
        except (sqlite3.Error, pd.errors.DatabaseError):
            frames = []

    if not frames:
        speeches_db = paths["speeches_catalog_db"]
        documents_db = paths["documents_catalog_db"]
        try:
            if speeches_db.exists() and _safe_table_exists(speeches_db, "speeches"):
                frames.append(
                    _read_sqlite_query(
                        speeches_db,
                        """
                        SELECT
                            'speech' AS source_type,
                            id AS doc_id,
                            title,
                            speaker,
                            event
                        FROM speeches
                        """,
                    )
                )
            if documents_db.exists() and _safe_table_exists(documents_db, "documents"):
                frames.append(
                    _read_sqlite_query(
                        documents_db,
                        """
                        SELECT
                            'document' AS source_type,
                            id AS doc_id,
                            COALESCE(meeting_label, category, doc_id) AS title,
                            NULL AS speaker,
                            category AS event
                        FROM documents
                        """,
                    )
                )
        except (sqlite3.Error, pd.errors.DatabaseError):
            return _empty_metadata_frame()

    if not frames:
        return _empty_metadata_frame()

    return _normalize_metadata_frame(pd.concat(frames, ignore_index=True))


def load_phase4_run_artifacts(
    *,
    run_version: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load predictions + metrics artifacts for a selected (or latest) phase4 run."""
    paths = _paths(repo_root)
    runs_root = paths["phase4_runs_root"]
    if run_version is None:
        run_version = latest_phase4_run(repo_root=repo_root, require_complete=True)
    if run_version is None:
        raise FileNotFoundError(
            f"No complete Phase 4 runs found under: {runs_root}. "
            f"Run: {remediation_commands()['phase4']}"
        )

    run_dir = runs_root / run_version
    if not run_dir.exists():
        raise FileNotFoundError(f"Requested run does not exist: {run_dir}")

    missing = missing_phase4_files(run_dir)
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Run {run_version!r} is incomplete. Missing: {missing_str}. "
            f"Rebuild with: {remediation_commands()['phase4']}"
        )

    predictions = _load_predictions(run_dir / "predictions.parquet")

    results_payload = json.loads((run_dir / "results_table.json").read_text(encoding="utf-8"))
    paired_payload = json.loads((run_dir / "paired_comparison.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    config_payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    return {
        "model_family": "SARIMAX",
        "model_key": "sarimax",
        "run_version": run_version,
        "run_dir": run_dir,
        "predictions": predictions.reset_index(drop=True),
        "results_table": pd.DataFrame(results_payload),
        "paired_comparison": pd.DataFrame(paired_payload),
        "run_summary": summary_payload,
        "run_config": config_payload,
        "feature_importance": pd.DataFrame(),
        "feature_schema": {},
        "model_path": None,
    }


def load_xgboost_run_artifacts(
    *,
    run_version: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load predictions, metrics, feature importance, and model metadata for an XGBoost run."""
    paths = _paths(repo_root)
    runs_root = paths["xgboost_runs_root"]
    if run_version is None:
        run_version = latest_xgboost_run(repo_root=repo_root, require_complete=True)
    if run_version is None:
        raise FileNotFoundError(
            f"No complete XGBoost runs found under: {runs_root}. "
            f"Run: {remediation_commands()['xgboost']}"
        )

    run_dir = runs_root / run_version
    if not run_dir.exists():
        raise FileNotFoundError(f"Requested XGBoost run does not exist: {run_dir}")

    missing = missing_xgboost_files(run_dir)
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"XGBoost run {run_version!r} is incomplete. Missing: {missing_str}. "
            f"Rebuild with: {remediation_commands()['xgboost']}"
        )

    predictions = _load_predictions(run_dir / "predictions.parquet")
    results_payload = json.loads((run_dir / "results_table.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    config_payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    importance_payload = json.loads((run_dir / "feature_importance.json").read_text(encoding="utf-8"))
    schema_payload = json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8"))

    return {
        "model_family": "XGBoost",
        "model_key": "xgboost",
        "run_version": run_version,
        "run_dir": run_dir,
        "predictions": predictions.reset_index(drop=True),
        "results_table": pd.DataFrame(results_payload),
        "paired_comparison": pd.DataFrame(),
        "run_summary": summary_payload,
        "run_config": config_payload,
        "feature_importance": pd.DataFrame(importance_payload),
        "feature_schema": schema_payload,
        "model_path": run_dir / "model.json",
    }


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    expected_cols = {"run_version", "model_variant", "split", "date", "actual", "pred"}
    missing_cols = expected_cols.difference(predictions.columns)
    if missing_cols:
        raise ValueError(f"predictions.parquet missing expected columns: {sorted(missing_cols)}")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize()
    return predictions.dropna(subset=["date"]).sort_values(
        ["split", "model_variant", "date"],
        kind="mergesort",
    )


def get_artifact_freshness(
    *,
    repo_root: Path | None = None,
    prefer_backfilled: bool = True,
) -> pd.DataFrame:
    """Build a table of artifact freshness timestamps."""
    paths = _paths(repo_root)
    feature_exists = True
    try:
        features_path = resolve_features_path(repo_root=repo_root, prefer_backfilled=prefer_backfilled)
    except FileNotFoundError:
        feature_exists = False
        features_path = paths["features_backfilled"]

    rows: list[dict[str, Any]] = [
        {"artifact": "doc_features", "path": str(features_path), "exists": feature_exists},
        {"artifact": "model_dataset", "path": str(paths["model_dataset"]), "exists": paths["model_dataset"].exists()},
        {"artifact": "time_splits", "path": str(paths["splits"]), "exists": paths["splits"].exists()},
        {"artifact": "phase4_runs_root", "path": str(paths["phase4_runs_root"]), "exists": paths["phase4_runs_root"].exists()},
        {
            "artifact": "xgboost_runs_root",
            "path": str(paths["xgboost_runs_root"]),
            "exists": paths["xgboost_runs_root"].exists(),
        },
    ]
    for row in rows:
        artifact_path = Path(row["path"])
        if row["exists"]:
            ts = datetime.fromtimestamp(artifact_path.stat().st_mtime, tz=timezone.utc)
            row["updated_at_utc"] = ts.replace(microsecond=0).isoformat()
        else:
            row["updated_at_utc"] = None
    return pd.DataFrame(rows)


def build_status_snapshot(
    *,
    repo_root: Path | None = None,
    prefer_backfilled: bool = True,
) -> dict[str, Any]:
    """Build readiness snapshot and guidance commands for dashboard pages."""
    paths = _paths(repo_root)
    commands = remediation_commands()

    selected_feature_path: Path | None = None
    try:
        selected_feature_path = resolve_features_path(repo_root=repo_root, prefer_backfilled=prefer_backfilled)
        features_exists = True
    except FileNotFoundError:
        features_exists = False

    model_dataset_exists = paths["model_dataset"].exists()
    splits_exists = paths["splits"].exists()

    all_runs = list_phase4_runs(repo_root=repo_root, require_complete=False)
    complete_runs = list_phase4_runs(repo_root=repo_root, require_complete=True)
    latest_any = all_runs[-1] if all_runs else None
    latest_complete = complete_runs[-1] if complete_runs else None
    xgboost_runs = list_xgboost_runs(repo_root=repo_root, require_complete=False)
    complete_xgboost_runs = list_xgboost_runs(repo_root=repo_root, require_complete=True)
    latest_xgboost = complete_xgboost_runs[-1] if complete_xgboost_runs else None

    latest_any_missing: list[str] = []
    if latest_any is not None and latest_complete != latest_any:
        latest_any_missing = missing_phase4_files(paths["phase4_runs_root"] / latest_any)
    latest_xgboost_missing: list[str] = []
    if xgboost_runs and latest_xgboost != xgboost_runs[-1]:
        latest_xgboost_missing = missing_xgboost_files(paths["xgboost_runs_root"] / xgboost_runs[-1])

    feature_path_text = (
        str(selected_feature_path)
        if selected_feature_path is not None
        else f"{paths['features_backfilled']} or {paths['features']}"
    )
    checks = [
        {
            "key": "features",
            "label": "Doc Feature Artifact",
            "path": feature_path_text,
            "ready": features_exists,
            "remediation_command": commands["features"],
        },
        {
            "key": "model_dataset",
            "label": "Model Dataset Artifact",
            "path": str(paths["model_dataset"]),
            "ready": model_dataset_exists,
            "remediation_command": commands["model_dataset"],
        },
        {
            "key": "splits",
            "label": "Time Split Artifact",
            "path": str(paths["splits"]),
            "ready": splits_exists,
            "remediation_command": commands["model_dataset"],
        },
        {
            "key": "phase4",
            "label": "Phase 4 Run Artifacts",
            "path": str(paths["phase4_runs_root"]),
            "ready": latest_complete is not None,
            "remediation_command": commands["phase4"],
        },
        {
            "key": "xgboost",
            "label": "XGBoost Run Artifacts",
            "path": str(paths["xgboost_runs_root"]),
            "ready": latest_xgboost is not None,
            "remediation_command": commands["xgboost"],
        },
    ]

    ready_count = sum(1 for item in checks if item["ready"])
    return {
        "checks": checks,
        "ready_count": ready_count,
        "total_checks": len(checks),
        "commands": commands,
        "selected_feature_path": str(selected_feature_path) if selected_feature_path is not None else None,
        "runs": {
            "all": all_runs,
            "complete": complete_runs,
            "latest_any": latest_any,
            "latest_complete": latest_complete,
            "latest_any_missing_files": latest_any_missing,
            "xgboost_all": xgboost_runs,
            "xgboost_complete": complete_xgboost_runs,
            "xgboost_latest_complete": latest_xgboost,
            "xgboost_latest_any_missing_files": latest_xgboost_missing,
        },
    }


__all__ = [
    "PHASE4_REQUIRED_FILES",
    "XGBOOST_REQUIRED_FILES",
    "remediation_commands",
    "resolve_features_path",
    "list_phase4_runs",
    "list_xgboost_runs",
    "latest_phase4_run",
    "latest_xgboost_run",
    "missing_phase4_files",
    "missing_xgboost_files",
    "load_features_dataframe",
    "load_model_dataset_dataframe",
    "load_time_splits_payload",
    "load_optional_document_metadata",
    "load_phase4_run_artifacts",
    "load_xgboost_run_artifacts",
    "get_artifact_freshness",
    "build_status_snapshot",
]

