from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from datasets.build_dataset.daily_alignment import (
    aggregate_communication_features_trailing_daily,
    build_daily_autoregressive_features,
)
import datasets.build_dataset.daily_builder as daily_builder_module
from datasets.build_dataset.daily_builder import BuildDailyDatasetConfig, build_daily_model_dataset


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _build_config(tmp_path: Path) -> BuildDailyDatasetConfig:
    return BuildDailyDatasetConfig(
        features_path=tmp_path / "data" / "features" / "doc_level" / "features.parquet",
        target_path=tmp_path / "data" / "targets" / "t5yie_diff1.parquet",
        output_dataset_path=tmp_path / "data" / "targets" / "model_dataset_t5yie_daily.parquet",
        split_output_path=tmp_path / "data" / "splits" / "time_splits_daily.json",
        summary_output_path=tmp_path / "reports" / "daily_dataset_summary.json",
        write_manifest_files=False,
    )


def _daily_feature_rows() -> list[dict]:
    return [
        {
            "doc_id": 1,
            "source_type": "speech",
            "date": "2016-12-30",
            "hawkish_score": 0.10,
            "novelty": 0.20,
            "n_target_sentences": 1,
            "target_sentences_ratio": 0.50,
            "text_length_words": 120,
            "role": "Chairman",
        },
        {
            "doc_id": 2,
            "source_type": "speech",
            "date": "2017-01-01",
            "hawkish_score": -0.20,
            "novelty": 0.40,
            "n_target_sentences": 2,
            "target_sentences_ratio": 0.40,
            "text_length_words": 95,
            "role": "Vice Chairman",
        },
        {
            "doc_id": 3,
            "source_type": "document",
            "date": "2020-12-31",
            "hawkish_score": 0.30,
            "novelty": 0.10,
            "n_target_sentences": 1,
            "target_sentences_ratio": 0.25,
            "text_length_words": 140,
            "role": None,
        },
        {
            # Future vs target date=2021-01-01. Must be excluded from 2021-01-01 windows.
            "doc_id": 4,
            "source_type": "speech",
            "date": "2021-01-02",
            "hawkish_score": 9.90,
            "novelty": 9.90,
            "n_target_sentences": 9,
            "target_sentences_ratio": 0.90,
            "text_length_words": 220,
            "role": "Chairman",
        },
    ]


def test_build_daily_autoregressive_features_shifted_no_lookahead():
    target_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="D"),
            "t5yie_diff1": list(range(1, 13)),
        }
    )
    out = build_daily_autoregressive_features(target_df, target_column="t5yie_diff1")

    assert out["t5yie_diff1_lag_1"].iloc[1] == 1
    assert out["t5yie_diff1_lag_2"].iloc[2] == 1
    assert out["t5yie_diff1_lag_5"].iloc[5] == 1
    assert out["t5yie_diff1_lag_10"].iloc[10] == 1
    assert pd.isna(out["t5yie_diff1_lag_1"].iloc[0])

    assert out["t5yie_diff1_roll_mean_5"].iloc[5] == pytest.approx(3.0)
    assert out["t5yie_diff1_roll_mean_10"].iloc[10] == pytest.approx(5.5)
    assert out["t5yie_diff1_roll_std_5"].iloc[5] == pytest.approx(pd.Series([1, 2, 3, 4, 5]).std())
    assert out["t5yie_diff1_roll_std_10"].iloc[10] == pytest.approx(pd.Series(range(1, 11)).std())

    pd.testing.assert_series_equal(
        out["t5yie_diff1_lag_1"],
        target_df["t5yie_diff1"].shift(1),
        check_names=False,
    )


def test_communication_windows_use_docs_up_to_target_date_only():
    features_df = pd.DataFrame(
        [
            {
                "doc_id": 1,
                "date": "2020-01-04",
                "hawkish_score": 0.1,
                "novelty": 0.2,
                "n_target_sentences": 1,
            },
            {
                "doc_id": 2,
                "date": "2020-01-10",
                "hawkish_score": -0.2,
                "novelty": 0.3,
                "n_target_sentences": 2,
            },
            {
                # Future row for t=2020-01-10; must be excluded.
                "doc_id": 3,
                "date": "2020-01-11",
                "hawkish_score": 9.0,
                "novelty": 0.9,
                "n_target_sentences": 5,
            },
        ]
    )
    features_df["date"] = pd.to_datetime(features_df["date"])

    out = aggregate_communication_features_trailing_daily(
        pd.to_datetime(["2020-01-10"]),
        features_df,
        window_days=(7,),
    )
    row = out.iloc[0]
    assert row["doc_count_7d"] == 2
    assert row["hawkish_score_sum_7d"] == pytest.approx(-0.1)
    assert row["n_target_sentences_sum_7d"] == pytest.approx(3.0)
    assert row["novelty_mean_7d"] == pytest.approx(0.25)


def test_communication_empty_window_semantics():
    features_df = pd.DataFrame(
        [
            {
                "doc_id": 1,
                "date": "2020-01-10",
                "hawkish_score": 0.4,
                "novelty": 0.5,
                "n_target_sentences": 2,
            }
        ]
    )
    features_df["date"] = pd.to_datetime(features_df["date"])

    out = aggregate_communication_features_trailing_daily(
        pd.to_datetime(["2020-01-01"]),
        features_df,
        window_days=(7,),
    )
    row = out.iloc[0]

    assert row["doc_count_7d"] == 0
    assert row["hawkish_score_sum_7d"] == pytest.approx(0.0)
    assert row["n_target_sentences_sum_7d"] == pytest.approx(0.0)
    assert pd.isna(row["hawkish_score_mean_7d"])
    assert pd.isna(row["novelty_mean_7d"])
    assert pd.isna(row["hawkish_score_max_abs_signed_7d"])


def test_event_feature_max_abs_signed_preserves_sign_and_tie_breaks_earliest():
    features_df = pd.DataFrame(
        [
            {
                "doc_id": 2,
                "date": "2020-01-08",
                "hawkish_score": 1.2,
                "novelty": 0.1,
                "n_target_sentences": 1,
            },
            {
                # Same abs value and same date; earliest doc_id should win.
                "doc_id": 1,
                "date": "2020-01-08",
                "hawkish_score": -1.2,
                "novelty": 0.2,
                "n_target_sentences": 1,
            },
            {
                "doc_id": 3,
                "date": "2020-01-09",
                "hawkish_score": 0.5,
                "novelty": 0.3,
                "n_target_sentences": 1,
            },
        ]
    )
    features_df["date"] = pd.to_datetime(features_df["date"])

    out = aggregate_communication_features_trailing_daily(
        pd.to_datetime(["2020-01-10"]),
        features_df,
        window_days=(7,),
    )
    row = out.iloc[0]
    assert row["hawkish_score_max_abs_signed_7d"] == pytest.approx(-1.2)


def test_build_daily_model_dataset_end_to_end_outputs_and_splits(tmp_path: Path):
    config = _build_config(tmp_path)

    target_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-12-31",
                    "2017-01-01",
                    "2020-12-31",
                    "2021-01-01",
                ]
            ),
            "t5yie_diff1": [0.10, 0.20, 0.30, 0.40],
        }
    )
    features_df = pd.DataFrame(_daily_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    out_path = build_daily_model_dataset(config)
    out_df = pd.read_parquet(out_path)
    split_payload = json.loads(config.split_output_path.read_text(encoding="utf-8"))

    assert out_path == config.output_dataset_path
    assert out_df["date"].is_monotonic_increasing
    assert out_df["date"].is_unique

    required_cols = {
        "date",
        "t5yie_diff1",
        "t5yie_diff1_lag_1",
        "t5yie_diff1_lag_2",
        "t5yie_diff1_lag_5",
        "t5yie_diff1_lag_10",
        "t5yie_diff1_roll_mean_5",
        "t5yie_diff1_roll_mean_10",
        "t5yie_diff1_roll_std_5",
        "t5yie_diff1_roll_std_10",
        "hawkish_score_max_abs_signed_7d",
        "doc_count_7d",
        "target_sentences_ratio_mean_7d",
        "text_length_words_max_7d",
        "role_share_chairman_7d",
    }
    assert required_cols.issubset(set(out_df.columns))

    pd.testing.assert_series_equal(
        out_df["t5yie_diff1_lag_1"],
        out_df["t5yie_diff1"].shift(1),
        check_names=False,
    )

    jan_2021_row = out_df.loc[out_df["date"] == pd.Timestamp("2021-01-01")].iloc[0]
    assert jan_2021_row["doc_count_7d"] == 1
    assert jan_2021_row["hawkish_score_sum_7d"] == pytest.approx(0.3)

    assert split_payload["counts"] == {"train": 1, "val": 2, "test": 1}
    assert split_payload["dates"]["train"] == ["2016-12-31"]
    assert split_payload["dates"]["val"] == ["2017-01-01", "2020-12-31"]
    assert split_payload["dates"]["test"] == ["2021-01-01"]


def test_main_cli_builds_daily_dataset_and_split_outputs(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-12-31",
                    "2017-01-01",
                    "2020-12-31",
                    "2021-01-01",
                ]
            ),
            "t5yie_diff1": [0.10, 0.20, 0.30, 0.40],
        }
    )
    features_df = pd.DataFrame(_daily_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    out_path = daily_builder_module.main(
        [
            "--features-path",
            str(config.features_path),
            "--target-path",
            str(config.target_path),
            "--output-dataset-path",
            str(config.output_dataset_path),
            "--split-output-path",
            str(config.split_output_path),
            "--summary-output-path",
            str(config.summary_output_path),
            "--no-manifest",
        ]
    )

    assert out_path == config.output_dataset_path
    assert config.output_dataset_path.exists()
    assert config.split_output_path.exists()
    assert config.summary_output_path is not None
    assert config.summary_output_path.exists()


def test_main_cli_parses_communication_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_build(config: BuildDailyDatasetConfig) -> Path:
        captured["config"] = config
        return config.output_dataset_path

    monkeypatch.setattr(daily_builder_module, "build_daily_model_dataset", _fake_build)

    out = daily_builder_module.main(
        [
            "--features-path",
            str(tmp_path / "features.parquet"),
            "--target-path",
            str(tmp_path / "target.parquet"),
            "--output-dataset-path",
            str(tmp_path / "out.parquet"),
            "--split-output-path",
            str(tmp_path / "splits.json"),
            "--communication-windows",
            "5,10,20",
            "--communication-lag-days",
            "2",
            "--no-manifest",
        ]
    )

    assert out == tmp_path / "out.parquet"
    config = captured["config"]
    assert isinstance(config, BuildDailyDatasetConfig)
    assert config.communication_windows == (5, 10, 20)
    assert config.communication_lag_days == 2
    assert config.write_manifest_files is False
