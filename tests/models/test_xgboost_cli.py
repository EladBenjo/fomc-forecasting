from __future__ import annotations

from pathlib import Path

import models.ml.xgboost as xgb_module


def test_main_cli_parses_expected_overrides(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run(config: xgb_module.XGBoostDailyConfig) -> Path:
        captured["config"] = config
        return config.output_root / "xgboost-run"

    monkeypatch.setattr(xgb_module, "run_xgboost_daily", _fake_run)

    dataset_path = Path("build") / "xgboost-cli-tests" / "dataset.parquet"
    split_path = Path("build") / "xgboost-cli-tests" / "splits.json"
    output_root = Path("build") / "xgboost-cli-tests" / "outputs"
    out = xgb_module.main(
        [
            "--dataset-path",
            str(dataset_path),
            "--split-path",
            str(split_path),
            "--output-root",
            str(output_root),
            "--run-version",
            "xgboost-custom",
            "--target-column",
            "custom_target",
            "--n-estimators",
            "12",
            "--max-depth",
            "4",
            "--learning-rate",
            "0.03",
            "--subsample",
            "0.8",
            "--colsample-bytree",
            "0.7",
            "--random-state",
            "123",
        ]
    )

    assert out == output_root / "xgboost-run"
    config = captured["config"]
    assert isinstance(config, xgb_module.XGBoostDailyConfig)
    assert config.dataset_path == dataset_path
    assert config.split_path == split_path
    assert config.output_root == output_root
    assert config.run_version == "xgboost-custom"
    assert config.target_column == "custom_target"
    assert config.n_estimators == 12
    assert config.max_depth == 4
    assert config.learning_rate == 0.03
    assert config.subsample == 0.8
    assert config.colsample_bytree == 0.7
    assert config.random_state == 123


def test_main_cli_parses_no_manifest(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run(config: xgb_module.XGBoostDailyConfig) -> Path:
        captured["config"] = config
        return Path("build") / "xgboost-cli-tests" / "out"

    monkeypatch.setattr(xgb_module, "run_xgboost_daily", _fake_run)

    out = xgb_module.main(
        [
            "--dataset-path",
            str(Path("build") / "xgboost-cli-tests" / "dataset.parquet"),
            "--split-path",
            str(Path("build") / "xgboost-cli-tests" / "splits.json"),
            "--output-root",
            str(Path("build") / "xgboost-cli-tests" / "outputs"),
            "--no-manifest",
        ]
    )

    assert out == Path("build") / "xgboost-cli-tests" / "out"
    config = captured["config"]
    assert isinstance(config, xgb_module.XGBoostDailyConfig)
    assert config.write_manifest_files is False
