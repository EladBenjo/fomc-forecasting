from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import models.baselines.sarimax as sarimax_module
from models.baselines.sarimax import SarimaxBenchmarkConfig, run_phase4_benchmark


def _write_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-12-01",
                    "2017-01-01",
                    "2017-02-01",
                    "2020-12-01",
                    "2021-01-01",
                    "2021-02-01",
                ]
            ),
            "t5yie_diff1": [0.1, 0.2, 0.3, 0.4, -0.1, 0.05],
            "hawkish_score": [0.1, 0.2, 0.2, 0.1, -0.1, 0.0],
            "novelty": [0.2, 0.3, 0.3, 0.4, 0.5, 0.6],
            "n_hawkish": [1, 2, 2, 1, 0, 1],
            "n_dovish": [0, 1, 1, 2, 2, 1],
            "n_target_sentences": [1, 3, 3, 3, 2, 2],
            "doc_count": [1, 2, 2, 2, 1, 1],
        }
    )
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


def _fake_runner(
    *,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    eval_start: pd.Timestamp | None,
    eval_end: pd.Timestamp | None,
    exog_cols: list[str],
    order: tuple[int, int, int],
    trend: str,
    min_train_obs: int,
) -> tuple[pd.DataFrame, int]:
    actual_map = dict(zip(df[date_col], df[target_col]))
    variant = (
        "baseline_univariate"
        if not exog_cols
        else "exog_minimal_counts"
        if exog_cols == ["hawkish_score", "novelty", "doc_count"]
        else "exog_share_variant"
    )
    if eval_start is not None and eval_start >= pd.Timestamp("2021-01-01"):
        dates = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]
    else:
        dates = [pd.Timestamp("2017-01-01"), pd.Timestamp("2020-12-01")]
        if variant == "exog_share_variant":
            dates = [pd.Timestamp("2020-12-01")]

    rows = []
    for dt in dates:
        actual = float(actual_map[dt])
        if variant == "baseline_univariate":
            pred = actual + 0.05
        elif variant == "exog_minimal_counts":
            pred = actual + 0.02
        else:
            pred = actual + 0.01
        rows.append({"date": dt, "actual": actual, "pred": pred})

    return pd.DataFrame(rows), 0


def test_run_phase4_benchmark_writes_expected_outputs(monkeypatch, tmp_path: Path):
    dataset_path = tmp_path / "data" / "targets" / "model_dataset_t5yie.parquet"
    split_path = tmp_path / "data" / "splits" / "time_splits.json"
    output_root = tmp_path / "data" / "models" / "baselines" / "t5yie"
    _write_dataset(dataset_path)
    _write_splits(split_path)
    monkeypatch.setattr(sarimax_module, "run_expanding_one_step_sarimax", _fake_runner)

    out_dir = run_phase4_benchmark(
        SarimaxBenchmarkConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="phase4-test-a",
            write_manifest_files=False,
        )
    )

    assert out_dir == output_root / "phase4-test-a"
    predictions_path = out_dir / "predictions.parquet"
    results_path = out_dir / "results_table.json"
    paired_path = out_dir / "paired_comparison.json"
    summary_path = out_dir / "run_summary.json"
    config_path = out_dir / "run_config.json"

    assert predictions_path.exists()
    assert results_path.exists()
    assert paired_path.exists()
    assert summary_path.exists()
    assert config_path.exists()

    results_rows = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(results_rows) == 6
    assert {row["model_variant"] for row in results_rows} == {
        "baseline_univariate",
        "exog_minimal_counts",
        "exog_share_variant",
    }
    paired_rows = json.loads(paired_path.read_text(encoding="utf-8"))
    val_share = next(
        row
        for row in paired_rows
        if row["split"] == "val" and row["exog_variant"] == "exog_share_variant"
    )
    assert val_share["n_common_dates"] == 1

    pred_df = pd.read_parquet(predictions_path)
    assert set(pred_df.columns) == {"run_version", "model_variant", "split", "date", "actual", "pred"}
    assert set(pred_df["run_version"]) == {"phase4-test-a"}


def test_run_phase4_benchmark_is_deterministic_across_versions(monkeypatch, tmp_path: Path):
    dataset_path = tmp_path / "data" / "targets" / "model_dataset_t5yie.parquet"
    split_path = tmp_path / "data" / "splits" / "time_splits.json"
    output_root = tmp_path / "data" / "models" / "baselines" / "t5yie"
    _write_dataset(dataset_path)
    _write_splits(split_path)
    monkeypatch.setattr(sarimax_module, "run_expanding_one_step_sarimax", _fake_runner)

    out_a = run_phase4_benchmark(
        SarimaxBenchmarkConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="phase4-test-a",
            write_manifest_files=False,
        )
    )
    out_b = run_phase4_benchmark(
        SarimaxBenchmarkConfig(
            dataset_path=dataset_path,
            split_path=split_path,
            output_root=output_root,
            run_version="phase4-test-b",
            write_manifest_files=False,
        )
    )

    res_a = json.loads((out_a / "results_table.json").read_text(encoding="utf-8"))
    res_b = json.loads((out_b / "results_table.json").read_text(encoding="utf-8"))
    assert res_a == res_b

    paired_a = json.loads((out_a / "paired_comparison.json").read_text(encoding="utf-8"))
    paired_b = json.loads((out_b / "paired_comparison.json").read_text(encoding="utf-8"))
    assert paired_a == paired_b

    pred_a = pd.read_parquet(out_a / "predictions.parquet").drop(columns=["run_version"]).reset_index(drop=True)
    pred_b = pd.read_parquet(out_b / "predictions.parquet").drop(columns=["run_version"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(pred_a, pred_b)

