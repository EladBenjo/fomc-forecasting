"""Streamlit-cached data loaders for dashboard pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.lib import artifacts


def _as_path(repo_root: str | None) -> Path | None:
    return Path(repo_root) if repo_root else None


@st.cache_data(show_spinner=False)
def get_status_snapshot(repo_root: str | None = None) -> dict[str, Any]:
    return artifacts.build_status_snapshot(repo_root=_as_path(repo_root))


@st.cache_data(show_spinner=False)
def get_artifact_freshness(repo_root: str | None = None) -> pd.DataFrame:
    return artifacts.get_artifact_freshness(repo_root=_as_path(repo_root))


@st.cache_data(show_spinner=False)
def get_features_frame(repo_root: str | None = None, prefer_backfilled: bool = True) -> pd.DataFrame:
    return artifacts.load_features_dataframe(
        repo_root=_as_path(repo_root),
        prefer_backfilled=prefer_backfilled,
    )


@st.cache_data(show_spinner=False)
def get_model_dataset_frame(repo_root: str | None = None) -> pd.DataFrame:
    return artifacts.load_model_dataset_dataframe(repo_root=_as_path(repo_root))


@st.cache_data(show_spinner=False)
def get_time_splits(repo_root: str | None = None) -> dict[str, Any]:
    return artifacts.load_time_splits_payload(repo_root=_as_path(repo_root))


@st.cache_data(show_spinner=False)
def get_phase4_runs(repo_root: str | None = None, require_complete: bool = True) -> list[str]:
    return artifacts.list_phase4_runs(
        repo_root=_as_path(repo_root),
        require_complete=require_complete,
    )


@st.cache_data(show_spinner=False)
def get_phase4_payload(run_version: str, repo_root: str | None = None) -> dict[str, Any]:
    return artifacts.load_phase4_run_artifacts(
        run_version=run_version,
        repo_root=_as_path(repo_root),
    )


@st.cache_data(show_spinner=False)
def get_latest_phase4_payload(repo_root: str | None = None) -> dict[str, Any] | None:
    latest = artifacts.latest_phase4_run(repo_root=_as_path(repo_root), require_complete=True)
    if latest is None:
        return None
    return artifacts.load_phase4_run_artifacts(run_version=latest, repo_root=_as_path(repo_root))


@st.cache_data(show_spinner=False)
def get_xgboost_runs(repo_root: str | None = None, require_complete: bool = True) -> list[str]:
    return artifacts.list_xgboost_runs(
        repo_root=_as_path(repo_root),
        require_complete=require_complete,
    )


@st.cache_data(show_spinner=False)
def get_xgboost_payload(run_version: str, repo_root: str | None = None) -> dict[str, Any]:
    return artifacts.load_xgboost_run_artifacts(
        run_version=run_version,
        repo_root=_as_path(repo_root),
    )


@st.cache_data(show_spinner=False)
def get_latest_xgboost_payload(repo_root: str | None = None) -> dict[str, Any] | None:
    latest = artifacts.latest_xgboost_run(repo_root=_as_path(repo_root), require_complete=True)
    if latest is None:
        return None
    return artifacts.load_xgboost_run_artifacts(run_version=latest, repo_root=_as_path(repo_root))


@st.cache_data(show_spinner=False)
def get_optional_metadata_frame(repo_root: str | None = None) -> pd.DataFrame:
    return artifacts.load_optional_document_metadata(repo_root=_as_path(repo_root))

