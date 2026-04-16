"""Read-only artifact access layer for the Streamlit MVP app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH_REL = Path("data/features/doc_level/features.parquet")
MODEL_DATASET_PATH_REL = Path("data/targets/model_dataset_t5yie.parquet")
SPLITS_PATH_REL = Path("data/splits/time_splits.json")
PHASE4_RUNS_ROOT_REL = Path("data/models/baselines/t5yie")

PHASE4_REQUIRED_FILES: tuple[str, ...] = (
    "predictions.parquet",
    "results_table.json",
    "paired_comparison.json",
    "run_summary.json",
    "run_config.json",
)


def remediation_commands() -> dict[str, str]:
    """Canonical CLI guidance for missing artifacts."""
    return {
        "features": "python -m fedtext.text.features.pipeline",
        "model_dataset": "python -m datasets.build_dataset.builder",
        "phase4": "python -m models.baselines.sarimax",
        "app": "streamlit run app/main.py",
    }


def _root(repo_root: Path | None = None) -> Path:
    return Path(repo_root) if repo_root is not None else REPO_ROOT


def _paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = _root(repo_root)
    return {
        "repo_root": root,
        "features": root / FEATURES_PATH_REL,
        "model_dataset": root / MODEL_DATASET_PATH_REL,
        "splits": root / SPLITS_PATH_REL,
        "phase4_runs_root": root / PHASE4_RUNS_ROOT_REL,
    }


def list_phase4_runs(
    *,
    repo_root: Path | None = None,
    require_complete: bool = False,
) -> list[str]:
    """List phase4 run versions under data/models/baselines/t5yie."""
    runs_root = _paths(repo_root)["phase4_runs_root"]
    if not runs_root.exists() or not runs_root.is_dir():
        return []

    discovered: list[str] = []
    for candidate in sorted(runs_root.iterdir(), key=lambda p: p.name):
        if not candidate.is_dir():
            continue
        missing = missing_phase4_files(candidate)
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


def missing_phase4_files(run_dir: Path) -> list[str]:
    """Return missing required files for a run directory."""
    return [name for name in PHASE4_REQUIRED_FILES if not (run_dir / name).exists()]


def load_features_dataframe(*, repo_root: Path | None = None) -> pd.DataFrame:
    """Load the doc-level feature artifact."""
    features_path = _paths(repo_root)["features"]
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature artifact: {features_path}")
    df = pd.read_parquet(features_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


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

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    expected_cols = {"run_version", "model_variant", "split", "date", "actual", "pred"}
    missing_cols = expected_cols.difference(predictions.columns)
    if missing_cols:
        raise ValueError(f"predictions.parquet missing expected columns: {sorted(missing_cols)}")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize()
    predictions = predictions.dropna(subset=["date"]).sort_values(
        ["split", "model_variant", "date"],
        kind="mergesort",
    )

    results_payload = json.loads((run_dir / "results_table.json").read_text(encoding="utf-8"))
    paired_payload = json.loads((run_dir / "paired_comparison.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    config_payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    return {
        "run_version": run_version,
        "run_dir": run_dir,
        "predictions": predictions.reset_index(drop=True),
        "results_table": pd.DataFrame(results_payload),
        "paired_comparison": pd.DataFrame(paired_payload),
        "run_summary": summary_payload,
        "run_config": config_payload,
    }


def build_status_snapshot(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Build readiness snapshot and guidance commands for the app status page."""
    paths = _paths(repo_root)
    commands = remediation_commands()

    features_exists = paths["features"].exists()
    model_dataset_exists = paths["model_dataset"].exists()
    splits_exists = paths["splits"].exists()

    all_runs = list_phase4_runs(repo_root=repo_root, require_complete=False)
    complete_runs = list_phase4_runs(repo_root=repo_root, require_complete=True)
    latest_any = all_runs[-1] if all_runs else None
    latest_complete = complete_runs[-1] if complete_runs else None

    latest_any_missing: list[str] = []
    if latest_any is not None and latest_complete != latest_any:
        latest_any_missing = missing_phase4_files(paths["phase4_runs_root"] / latest_any)

    checks = [
        {
            "key": "features",
            "label": "Feature Artifact",
            "path": str(paths["features"]),
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
    ]

    ready_count = sum(1 for item in checks if item["ready"])
    return {
        "checks": checks,
        "ready_count": ready_count,
        "total_checks": len(checks),
        "commands": commands,
        "runs": {
            "all": all_runs,
            "complete": complete_runs,
            "latest_any": latest_any,
            "latest_complete": latest_complete,
            "latest_any_missing_files": latest_any_missing,
        },
    }


__all__ = [
    "PHASE4_REQUIRED_FILES",
    "remediation_commands",
    "list_phase4_runs",
    "latest_phase4_run",
    "missing_phase4_files",
    "load_features_dataframe",
    "load_phase4_run_artifacts",
    "build_status_snapshot",
]

