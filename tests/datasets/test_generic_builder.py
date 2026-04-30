from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasets.build_dataset.feature_generators import build_target_features, build_trailing_exogenous_features
from datasets.build_dataset.generic_builder import build_generic_model_dataset


def test_target_features_lag_roll_momentum():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=25, freq="D"), "y": range(25)})
    out, cols = build_target_features(df, date_col="date", target_col="y", lags=[1, 2, 5], rolling_windows=[5], momentum_pairs=[(1, 5)])
    assert "y_lag_1" in cols
    assert out.loc[5, "y_lag_1"] == 4
    assert out.loc[5, "y_lag_5"] == 0
    assert out.loc[5, "y_momentum_lag_1_5"] == 4


def test_trailing_windows_no_lookahead_and_missing_flags():
    t = pd.to_datetime(["2020-01-10"])
    exog = pd.DataFrame({"date": pd.to_datetime(["2020-01-09", "2020-01-11"]), "hawkish_score": [1.0, 9.0]})
    out, cols = build_trailing_exogenous_features(t, exog, target_date_col="date", exog_date_col="date", exog_feature_columns=["hawkish_score"], windows=[7], aggregations=["sum", "count"], lag_days=0)
    assert out.loc[0, "hawkish_score_sum_7d"] == 1.0
    assert out.loc[0, "has_text_signal_7d"] == 1
    assert "event_count_7d" in cols


def test_generic_builder_writes_dataset_split_manifest_registry(tmp_path: Path):
    target = pd.DataFrame({"date": pd.date_range("2016-12-25", periods=15, freq="D"), "t5yie_diff1": range(15)})
    exog = pd.DataFrame({"date": pd.date_range("2016-12-20", periods=20, freq="D"), "hawkish_score": range(20), "novelty": range(20)})
    target_path = tmp_path / "data/targets/t5yie_diff1.parquet"
    exog_path = tmp_path / "data/features/doc_level/features.parquet"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    exog_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_parquet(target_path, index=False)
    exog.to_parquet(exog_path, index=False)

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(f"""
target:
  path: {target_path}
  value_column: t5yie_diff1
  date_column: date
features:
  target_lags: [1]
  target_rolling_windows: [5]
  exogenous_tables:
    - path: {exog_path}
      date_column: date
      feature_columns: [hawkish_score, novelty]
      windows: [7]
      aggregations: [mean, sum, count]
splits:
  train_end: '2016-12-31'
  val_start: '2017-01-01'
  val_end: '2017-01-03'
  test_start: '2017-01-04'
outputs:
  dataset_path: {tmp_path / 'data/targets/model_dataset_t5yie_generic.parquet'}
  split_path: {tmp_path / 'data/splits/time_splits_t5yie_generic.json'}
  manifest_out_dir: {tmp_path / 'data/targets'}
  registry_path: {tmp_path / 'data/targets/model_dataset_registry.sqlite3'}
""", encoding="utf-8")
    out = build_generic_model_dataset(config_path)
    assert out.exists()
    assert (tmp_path / "data/splits/time_splits_t5yie_generic.json").exists()
    payload = json.loads((tmp_path / "data/splits/time_splits_t5yie_generic.json").read_text())
    assert payload["counts"]["train"] > 0
    manifests = list((tmp_path / "data/targets/manifests").glob("*.json"))
    assert manifests
