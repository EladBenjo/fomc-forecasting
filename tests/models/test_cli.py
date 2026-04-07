from __future__ import annotations

from pathlib import Path

import models.baselines.sarimax as sarimax_module


def test_main_cli_parses_expected_overrides(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_run(config: sarimax_module.SarimaxBenchmarkConfig) -> Path:
        captured["config"] = config
        return config.output_root / "phase4-run"

    monkeypatch.setattr(sarimax_module, "run_phase4_benchmark", _fake_run)

    dataset_path = tmp_path / "dataset.parquet"
    split_path = tmp_path / "splits.json"
    output_root = tmp_path / "outputs"
    out = sarimax_module.main(
        [
            "--dataset-path",
            str(dataset_path),
            "--split-path",
            str(split_path),
            "--output-root",
            str(output_root),
            "--run-version",
            "phase4-custom",
            "--target-column",
            "t5yie_diff1",
            "--model-order",
            "2,0,1",
            "--model-trend",
            "n",
            "--min-train-obs",
            "24",
            "--meaningful-threshold-pct",
            "3.5",
            "--manifest-out-dir",
            str(tmp_path / "manifest"),
            "--manifest-registry-path",
            str(tmp_path / "manifest" / "registry.sqlite3"),
        ]
    )

    assert out == output_root / "phase4-run"
    config = captured["config"]
    assert isinstance(config, sarimax_module.SarimaxBenchmarkConfig)
    assert config.dataset_path == dataset_path
    assert config.split_path == split_path
    assert config.output_root == output_root
    assert config.run_version == "phase4-custom"
    assert config.model_order == (2, 0, 1)
    assert config.model_trend == "n"
    assert config.min_train_obs == 24
    assert config.meaningful_threshold_pct == 3.5
    assert config.manifest_out_dir == tmp_path / "manifest"
    assert config.manifest_registry_path == tmp_path / "manifest" / "registry.sqlite3"


def test_main_cli_parses_no_manifest(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_run(config: sarimax_module.SarimaxBenchmarkConfig) -> Path:
        captured["config"] = config
        return tmp_path / "out"

    monkeypatch.setattr(sarimax_module, "run_phase4_benchmark", _fake_run)

    out = sarimax_module.main(
        [
            "--dataset-path",
            str(tmp_path / "dataset.parquet"),
            "--split-path",
            str(tmp_path / "splits.json"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--no-manifest",
        ]
    )

    assert out == tmp_path / "out"
    config = captured["config"]
    assert isinstance(config, sarimax_module.SarimaxBenchmarkConfig)
    assert config.write_manifest_files is False

