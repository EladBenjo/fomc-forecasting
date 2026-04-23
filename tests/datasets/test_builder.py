from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from datasets.build_dataset.alignment import aggregate_features_monthly
import datasets.build_dataset.builder as builder_module
from datasets.build_dataset import BuildDatasetConfig, build_model_dataset
from datasets.schema.fields import MODEL_DATASET_COLUMNS


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_base_feature_rows() -> list[dict]:
    return [
        {
            "doc_id": 1,
            "source_type": "document",
            "date": "2016-12-05",
            "hawkish_score": 1.0,
            "n_hawkish": 1,
            "n_dovish": 0,
            "n_neutral": 0,
            "n_target_sentences": 1,
            "novelty": 0.20,
            "text_length_words": 120,
            "role": None,
        },
        {
            "doc_id": 2,
            "source_type": "speech",
            "date": "2016-12-20",
            "hawkish_score": 0.0,
            "n_hawkish": 0,
            "n_dovish": 0,
            "n_neutral": 1,
            "n_target_sentences": 1,
            "novelty": 0.10,
            "text_length_words": 80,
            "role": "Chair-Man",
        },
        {
            "doc_id": 3,
            "source_type": "speech",
            "date": "2017-01-10",
            "hawkish_score": -1.0,
            "n_hawkish": 0,
            "n_dovish": 1,
            "n_neutral": 0,
            "n_target_sentences": 1,
            "novelty": 0.30,
            "text_length_words": 95,
            "role": "Vice Chairman",
        },
        {
            "doc_id": 4,
            "source_type": "document",
            "date": "2017-03-12",
            "hawkish_score": 0.5,
            "n_hawkish": 1,
            "n_dovish": 0,
            "n_neutral": 0,
            "n_target_sentences": 1,
            "novelty": 0.40,
            "text_length_words": 140,
            "role": None,
        },
    ]


def _make_alignment_target_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2017-01-03",
                    "2017-01-20",
                    "2017-02-01",
                    "2017-02-20",
                    "2017-03-01",
                    "2017-03-20",
                    "2017-04-01",
                    "2017-04-20",
                ]
            ),
            "t5yie_diff1": [0.10, 0.30, -0.20, 0.00, 0.50, 0.10, -0.10, 0.10],
        }
    )


def _build_config(tmp_path: Path) -> BuildDatasetConfig:
    return BuildDatasetConfig(
        features_path=tmp_path / "data" / "features" / "doc_level" / "features.parquet",
        target_path=tmp_path / "data" / "targets" / "t5yie_diff1.parquet",
        output_dataset_path=tmp_path / "data" / "targets" / "model_dataset_t5yie.parquet",
        split_output_path=tmp_path / "data" / "splits" / "time_splits.json",
        summary_output_path=tmp_path / "reports" / "dataset_summary.json",
    )


def test_build_model_dataset_monthly_alignment_and_missing_semantics(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    out_path = build_model_dataset(config)
    out_df = pd.read_parquet(out_path)

    assert out_path == config.output_dataset_path
    assert tuple(out_df.columns) == MODEL_DATASET_COLUMNS
    assert out_df["target_month"].tolist() == ["2017-01", "2017-02", "2017-03", "2017-04"]
    assert out_df["feature_month_used"].tolist() == ["2016-12", "2017-01", "2017-02", "2017-03"]
    assert pd.to_datetime(out_df["date"]).is_monotonic_increasing

    # 2017-03 uses 2017-02 feature month, which has no docs -> NaNs preserved.
    march_row = out_df.loc[out_df["target_month"] == "2017-03"].iloc[0]
    assert pd.isna(march_row["hawkish_score"])
    assert pd.isna(march_row["text_length_words_max"])
    assert pd.isna(march_row["role_share_chairman"])
    assert pd.isna(march_row["hawkish_score_max_abs_signed_30d"])
    assert march_row["missing_period_reason"] == "no_docs_month"

    jan_row = out_df.loc[out_df["target_month"] == "2017-01"].iloc[0]
    assert jan_row["text_length_words_max"] == 120
    assert jan_row["role_share_chairman"] == pytest.approx(1.0)
    assert pd.isna(jan_row["hawkish_score_max_abs_signed_7d"])
    assert jan_row["hawkish_score_max_abs_signed_14d"] == pytest.approx(0.0)
    assert jan_row["hawkish_score_max_abs_signed_30d"] == pytest.approx(1.0)

    feb_row = out_df.loc[out_df["target_month"] == "2017-02"].iloc[0]
    assert feb_row["text_length_words_max"] == 95
    assert feb_row["role_share_chairman"] == pytest.approx(0.0)
    assert pd.isna(feb_row["hawkish_score_max_abs_signed_7d"])
    assert pd.isna(feb_row["hawkish_score_max_abs_signed_14d"])
    assert feb_row["hawkish_score_max_abs_signed_30d"] == pytest.approx(-1.0)

    # No-lookahead: feature month is always strictly earlier than target month.
    target_month = pd.PeriodIndex(out_df["target_month"], freq="M")
    feature_month_used = pd.PeriodIndex(out_df["feature_month_used"], freq="M")
    assert (feature_month_used < target_month).all()


def test_build_model_dataset_is_deterministic(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    first_path = build_model_dataset(config)
    first_df = pd.read_parquet(first_path)
    split_path = config.split_output_path
    first_split = json.loads(split_path.read_text(encoding="utf-8"))

    second_path = build_model_dataset(config)
    second_df = pd.read_parquet(second_path)
    second_split = json.loads(split_path.read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(first_df, second_df)
    assert first_split["boundaries"] == second_split["boundaries"]
    assert first_split["counts"] == second_split["counts"]
    assert first_split["dates"] == second_split["dates"]


def test_build_model_dataset_fixed_split_boundaries(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-12-05",
                    "2016-12-20",
                    "2017-01-05",
                    "2017-01-20",
                    "2020-12-05",
                    "2020-12-20",
                    "2021-01-05",
                    "2021-01-20",
                ]
            ),
            "t5yie_diff1": [0.1, 0.2, 0.0, 0.1, -0.2, -0.1, 0.3, 0.4],
        }
    )
    features_df = pd.DataFrame(
        [
            {
                "doc_id": 1,
                "source_type": "speech",
                "date": "2016-11-15",
                "hawkish_score": 0.2,
                "n_hawkish": 1,
                "n_dovish": 0,
                "n_neutral": 0,
                "n_target_sentences": 1,
                "novelty": 0.2,
                "text_length_words": 100,
                "role": "Chairman",
            },
            {
                "doc_id": 2,
                "source_type": "speech",
                "date": "2016-12-15",
                "hawkish_score": 0.3,
                "n_hawkish": 1,
                "n_dovish": 0,
                "n_neutral": 0,
                "n_target_sentences": 1,
                "novelty": 0.2,
                "text_length_words": 90,
                "role": "Vice Chairman",
            },
            {
                "doc_id": 3,
                "source_type": "speech",
                "date": "2020-11-15",
                "hawkish_score": -0.1,
                "n_hawkish": 0,
                "n_dovish": 1,
                "n_neutral": 0,
                "n_target_sentences": 1,
                "novelty": 0.3,
                "text_length_words": 75,
                "role": "Governor",
            },
            {
                "doc_id": 4,
                "source_type": "speech",
                "date": "2020-12-15",
                "hawkish_score": -0.2,
                "n_hawkish": 0,
                "n_dovish": 1,
                "n_neutral": 0,
                "n_target_sentences": 1,
                "novelty": 0.3,
                "text_length_words": 85,
                "role": "Chairman",
            },
        ]
    )
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    build_model_dataset(config)
    split_payload = json.loads(config.split_output_path.read_text(encoding="utf-8"))
    assert split_payload["counts"] == {"train": 1, "val": 2, "test": 1}
    assert split_payload["dates"]["train"] == ["2016-12-01"]
    assert split_payload["dates"]["val"] == ["2017-01-01", "2020-12-01"]
    assert split_payload["dates"]["test"] == ["2021-01-01"]


def test_build_model_dataset_fails_on_missing_required_columns(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows()).drop(columns=["novelty"])
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    with pytest.raises(
        ValueError,
        match="missing required input columns for selected monthly features",
    ):
        build_model_dataset(config)


def test_build_model_dataset_fails_on_non_monotonic_target_dates(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = _make_alignment_target_df().sample(frac=1.0, random_state=42).reset_index(drop=True)
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    with pytest.raises(ValueError, match="monotonic increasing"):
        build_model_dataset(config)


def test_build_model_dataset_fails_on_invalid_split_config(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    bad_config = BuildDatasetConfig(
        features_path=config.features_path,
        target_path=config.target_path,
        output_dataset_path=config.output_dataset_path,
        split_output_path=config.split_output_path,
        train_end="2017-01-01",
        val_start="2017-01-01",
        val_end="2020-12-31",
        test_start="2021-01-01",
    )

    with pytest.raises(ValueError, match="train_end must be before val_start"):
        build_model_dataset(bad_config)


def test_aggregate_features_monthly_event_features_tie_break_and_empty_windows():
    features_df = pd.DataFrame(
        [
            {"doc_id": 1, "date": "2020-01-30", "hawkish_score": 0.9},
            {"doc_id": 2, "date": "2020-01-31", "hawkish_score": -0.9},
            {"doc_id": 3, "date": "2020-01-31", "hawkish_score": 0.9},
            {"doc_id": 4, "date": "2020-01-05", "hawkish_score": -0.4},
            {"doc_id": 5, "date": "2020-02-01", "hawkish_score": -0.7},
            {"doc_id": 6, "date": "2020-02-10", "hawkish_score": 0.2},
        ]
    )
    features_df["date"] = pd.to_datetime(features_df["date"])

    monthly = aggregate_features_monthly(
        features_df,
        monthly_feature_columns=(
            "hawkish_score_max_abs_signed_7d",
            "hawkish_score_max_abs_signed_14d",
            "hawkish_score_max_abs_signed_30d",
        ),
    )

    jan = monthly.loc[monthly["feature_month"] == pd.Period("2020-01", freq="M")].iloc[0]
    assert jan["hawkish_score_max_abs_signed_7d"] == pytest.approx(0.9)
    assert jan["hawkish_score_max_abs_signed_14d"] == pytest.approx(0.9)
    assert jan["hawkish_score_max_abs_signed_30d"] == pytest.approx(0.9)

    feb = monthly.loc[monthly["feature_month"] == pd.Period("2020-02", freq="M")].iloc[0]
    assert pd.isna(feb["hawkish_score_max_abs_signed_7d"])
    assert pd.isna(feb["hawkish_score_max_abs_signed_14d"])
    assert feb["hawkish_score_max_abs_signed_30d"] == pytest.approx(-0.7)


def test_build_model_dataset_selected_feature_dependencies_and_legacy_subset_compat(tmp_path: Path):
    config = _build_config(tmp_path)
    config.write_manifest_files = False

    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows()).drop(columns=["text_length_words", "role"])
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    with pytest.raises(
        ValueError,
        match="missing required input columns for selected monthly features",
    ):
        build_model_dataset(config)

    legacy_columns = (
        "hawkish_score",
        "novelty",
        "n_hawkish",
        "n_dovish",
        "n_neutral",
        "n_target_sentences",
        "doc_count",
    )
    config.monthly_feature_columns = legacy_columns

    out_path = build_model_dataset(config)
    out_df = pd.read_parquet(out_path)
    expected_columns = (
        "target_month",
        "date",
        "feature_month_used",
        "t5yie_diff1",
        "t5yie_diff1_abs_mean",
        "t5yie_obs",
        *legacy_columns,
        "missing_period_reason",
    )
    assert tuple(out_df.columns) == expected_columns


def test_main_cli_builds_dataset_and_split_outputs(tmp_path: Path):
    config = _build_config(tmp_path)
    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])

    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    out_path = builder_module.main(
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
        ]
    )

    assert out_path == config.output_dataset_path
    assert config.output_dataset_path.exists()
    assert config.split_output_path.exists()
    assert config.summary_output_path is not None
    assert config.summary_output_path.exists()


def test_main_cli_parses_expected_rows_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_build(config: BuildDatasetConfig) -> Path:
        captured["config"] = config
        return config.output_dataset_path

    monkeypatch.setattr(builder_module, "build_model_dataset", _fake_build)

    features_path = tmp_path / "features.parquet"
    target_path = tmp_path / "target.parquet"
    output_path = tmp_path / "out.parquet"
    split_path = tmp_path / "splits.json"
    expected_json = '{"dataset_rows": 10, "target_rows": 20}'
    out = builder_module.main(
        [
            "--features-path",
            str(features_path),
            "--target-path",
            str(target_path),
            "--output-dataset-path",
            str(output_path),
            "--split-output-path",
            str(split_path),
            "--expected-rows-json",
            expected_json,
        ]
    )

    assert out == output_path
    config = captured["config"]
    assert isinstance(config, BuildDatasetConfig)
    assert config.expected_rows == {"dataset_rows": 10, "target_rows": 20}


def test_build_model_dataset_writes_manifest_and_registry_by_default(tmp_path: Path):
    config = _build_config(tmp_path)
    config.dataset_version = "model-dataset-t5yie-test"

    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])
    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    build_model_dataset(config)

    manifest_path = (
        config.output_dataset_path.parent / "manifests" / "model-dataset-t5yie-test.json"
    )
    registry_path = config.output_dataset_path.parent / "model_dataset_registry.sqlite3"

    assert manifest_path.exists()
    assert registry_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "model-dataset-t5yie-test"
    assert payload["target_column"] == "t5yie_diff1"
    assert payload["output_dataset_rows"] == 4

    conn = sqlite3.connect(registry_path)
    try:
        row = conn.execute(
            """
            SELECT dataset_version, target_column, output_dataset_rows
            FROM dataset_registry
            WHERE dataset_version = 'model-dataset-t5yie-test'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row == ("model-dataset-t5yie-test", "t5yie_diff1", 4)


def test_build_model_dataset_no_manifest_when_disabled(tmp_path: Path):
    config = _build_config(tmp_path)
    config.write_manifest_files = False

    target_df = _make_alignment_target_df()
    features_df = pd.DataFrame(_make_base_feature_rows())
    features_df["date"] = pd.to_datetime(features_df["date"])
    _write_parquet(target_df, config.target_path)
    _write_parquet(features_df, config.features_path)

    build_model_dataset(config)

    assert not (config.output_dataset_path.parent / "model_dataset_registry.sqlite3").exists()
    assert not (config.output_dataset_path.parent / "manifests").exists()


def test_main_cli_parses_no_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_build(config: BuildDatasetConfig) -> Path:
        captured["config"] = config
        return config.output_dataset_path

    monkeypatch.setattr(builder_module, "build_model_dataset", _fake_build)

    out = builder_module.main(
        [
            "--features-path",
            str(tmp_path / "features.parquet"),
            "--target-path",
            str(tmp_path / "target.parquet"),
            "--output-dataset-path",
            str(tmp_path / "out.parquet"),
            "--split-output-path",
            str(tmp_path / "splits.json"),
            "--no-manifest",
        ]
    )
    assert out == tmp_path / "out.parquet"

    config = captured["config"]
    assert isinstance(config, BuildDatasetConfig)
    assert config.write_manifest_files is False
