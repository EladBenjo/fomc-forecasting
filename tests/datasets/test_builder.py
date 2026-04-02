from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
    assert march_row["missing_period_reason"] == "no_docs_month"

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

    with pytest.raises(ValueError, match="missing required columns"):
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

