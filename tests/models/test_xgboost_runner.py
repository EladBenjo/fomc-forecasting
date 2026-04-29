from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import models.ml.xgboost as xgb_module
from models.ml.xgboost import XGBoostDailyConfig, run_xgboost_daily


TEST_WORK_ROOT = Path("build") / "xgboost-runner-tests"


class _FakeXGBRegressor:
    def __init__(self, **params: Any) -> None:
        self.params = params
        self.feature_importances_: np.ndarray | None = None
        self.mean_: float = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_FakeXGBRegressor":
        self.mean_ = float(y.mean())
        weights = np.arange(1, len(x.columns) + 1, dtype=float)
        self.feature_importances_ = weights / weights.sum()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.mean_, dtype=float)

    def save_model(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"mean": self.mean_, "params": self.params}, sort_keys=True),
            encoding="utf-8",
        )


def _write_dataset(path: Path) -> None:
    dates = pd.to_datetime(
        [
            "2016-12-29",
            "2016-12-30",
            "2017-01-03",
            "2017-01-04",
            "2020-12-30",
            "2020-12-31",
            "2021-01-04",
            "2021-01-05",
        ]
    )
    df = pd.DataFrame(
        {
            "date": dates,
            "t5yie_diff1": [0.10, 0.20, 0.05, -0.02, 0.30, 0.15, -0.10, 0.04],
        }
    )
    for i, col in enumerate(xgb_module.FEATURE_COLUMNS, start=1):
        df[col] = np.linspace(0.01 * i, 0.01 * i + 0.07, len(df))
    df.loc[2, "hawkish_score_mean_7d"] = np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_splits(path: Path) -> None:
    payload = {
        "boundaries": {
            "train_end": "2016-12-31",
            "val_start": "2017-01-01",
            "val_end": "2020-12-31",
            "test_start": "2021-01-01",
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def work_dir(request: pytest.FixtureRequest) -> Path:
    path = TEST_WORK_ROOT / request.node.name
    if path.exists():
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_default_config_uses_daily_artifacts_and_xgboost_output_root():
    config = XGBoostDailyConfig()

    assert config.dataset_path == xgb_module.DEFAULT_DATASET_PATH
    assert config.split_path == xgb_module.DEFAULT_SPLITS_PATH
    assert config.output_root == xgb_module.DEFAULT_OUTPUT_ROOT
    assert config.dataset_path.name == "model_dataset_t5yie_daily.parquet"
    assert config.split_path.name == "time_splits_daily.json"
    assert config.output_root.parts[-3:] == ("models", "xgboost", "t5yie")


def test_run_xgboost_daily_writes_expected_outputs(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
):
    dataset_path = work_dir / "data" / "targets" / "model_dataset_t5yie_daily.parquet"
    split_path = work_dir / "data" / "splits" / "time_splits_daily.json"
    output_root = work_dir / "data" / "models" / "xgboost" / "t5yie"
    _write_dataset(dataset_path)
    _write_splits(split_path)
    monkeypatch.setattr(xgb_module, "_load_xgb_regressor", lambda: _FakeXGBRegressor)

    out_dir = run_xgboost_daily(
        XGBoostDailyConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="xgboost-test-a",
            n_estimators=5,
            write_manifest_files=False,
        )
    )

    assert out_dir == output_root / "xgboost-test-a"
    expected_files = {
        "predictions.parquet",
        "results_table.json",
        "run_summary.json",
        "run_config.json",
        "feature_importance.json",
        "model.json",
        "feature_schema.json",
    }
    assert {path.name for path in out_dir.iterdir()} == expected_files

    pred_df = pd.read_parquet(out_dir / "predictions.parquet")
    assert set(pred_df.columns) == {"run_version", "model_variant", "split", "date", "actual", "pred"}
    assert set(pred_df["run_version"]) == {"xgboost-test-a"}
    assert set(pred_df["split"]) == {"val", "test"}
    assert set(pred_df["model_variant"]) == {xgb_module.MODEL_VARIANT}

    results_rows = json.loads((out_dir / "results_table.json").read_text(encoding="utf-8"))
    assert len(results_rows) == 2
    assert {row["split"] for row in results_rows} == {"val", "test"}
    assert {row["model_variant"] for row in results_rows} == {xgb_module.MODEL_VARIANT}
    assert all(row["n_forecasts"] > 0 for row in results_rows)
    assert all(row["feature_count"] == len(xgb_module.FEATURE_COLUMNS) for row in results_rows)

    config_payload = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config_payload["dataset_path"] == str(dataset_path)
    assert config_payload["split_path"] == str(split_path)
    assert config_payload["target_column"] == "t5yie_diff1"
    assert config_payload["model_variant"] == xgb_module.MODEL_VARIANT
    assert config_payload["feature_columns"] == list(xgb_module.FEATURE_COLUMNS)
    assert config_payload["xgboost_params"]["n_estimators"] == 5

    importance = json.loads((out_dir / "feature_importance.json").read_text(encoding="utf-8"))
    assert [row["feature"] for row in importance] == list(xgb_module.FEATURE_COLUMNS)

    schema = json.loads((out_dir / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["feature_columns"] == list(xgb_module.FEATURE_COLUMNS)
    assert schema["missing_rate"]["hawkish_score_mean_7d"] > 0


def test_run_xgboost_daily_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
):
    dataset_path = work_dir / "data" / "targets" / "model_dataset_t5yie_daily.parquet"
    split_path = work_dir / "data" / "splits" / "time_splits_daily.json"
    output_root = work_dir / "data" / "models" / "xgboost" / "t5yie"
    _write_dataset(dataset_path)
    _write_splits(split_path)
    monkeypatch.setattr(xgb_module, "_load_xgb_regressor", lambda: _FakeXGBRegressor)

    run_xgboost_daily(
        XGBoostDailyConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="xgboost-test-manifest",
        )
    )

    manifest_path = work_dir / "data" / "models" / "xgboost" / "manifests" / "xgboost-test-manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_version"] == "xgboost-test-manifest"
    assert payload["model_family"] == "xgboost"
    assert payload["outputs"]["predictions"]["rows"] > 0
    assert payload["outputs"]["model"]["sha256"]


def test_run_xgboost_daily_fails_on_missing_daily_feature_columns(work_dir: Path):
    dataset_path = work_dir / "data" / "targets" / "model_dataset_t5yie_daily.parquet"
    split_path = work_dir / "data" / "splits" / "time_splits_daily.json"
    output_root = work_dir / "data" / "models" / "xgboost" / "t5yie"
    _write_dataset(dataset_path)
    _write_splits(split_path)

    df = pd.read_parquet(dataset_path).drop(columns=["hawkish_score_sum_30d"])
    df.to_parquet(dataset_path, index=False)

    with pytest.raises(
        ValueError,
        match=(
            "Daily model dataset missing required columns for XGBoost.*"
            "python -m datasets.build_dataset.daily_builder"
        ),
    ):
        run_xgboost_daily(
            XGBoostDailyConfig(
                dataset_path=dataset_path,
                split_path=split_path,
                output_root=output_root,
                run_version="xgboost-test-missing-cols",
                write_manifest_files=False,
            )
        )


def test_run_xgboost_daily_is_deterministic_across_versions(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
):
    dataset_path = work_dir / "data" / "targets" / "model_dataset_t5yie_daily.parquet"
    split_path = work_dir / "data" / "splits" / "time_splits_daily.json"
    output_root = work_dir / "data" / "models" / "xgboost" / "t5yie"
    _write_dataset(dataset_path)
    _write_splits(split_path)
    monkeypatch.setattr(xgb_module, "_load_xgb_regressor", lambda: _FakeXGBRegressor)

    out_a = run_xgboost_daily(
        XGBoostDailyConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="xgboost-test-a",
            random_state=7,
            write_manifest_files=False,
        )
    )
    out_b = run_xgboost_daily(
        XGBoostDailyConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="xgboost-test-b",
            random_state=7,
            write_manifest_files=False,
        )
    )

    res_a = json.loads((out_a / "results_table.json").read_text(encoding="utf-8"))
    res_b = json.loads((out_b / "results_table.json").read_text(encoding="utf-8"))
    assert res_a == res_b

    summary_a = json.loads((out_a / "run_summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((out_b / "run_summary.json").read_text(encoding="utf-8"))
    assert summary_a == summary_b

    config_a = json.loads((out_a / "run_config.json").read_text(encoding="utf-8"))
    config_b = json.loads((out_b / "run_config.json").read_text(encoding="utf-8"))
    config_a.pop("run_version")
    config_b.pop("run_version")
    assert config_a == config_b

    pred_a = pd.read_parquet(out_a / "predictions.parquet").drop(columns=["run_version"]).reset_index(drop=True)
    pred_b = pd.read_parquet(out_b / "predictions.parquet").drop(columns=["run_version"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(pred_a, pred_b)
