"""Config-driven generic daily time-series dataset builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.build_dataset.alignment import normalize_dates
from datasets.build_dataset.feature_generators import build_target_features, build_trailing_exogenous_features
from datasets.build_dataset.versioning import ModelDatasetManifest, default_dataset_version, get_git_sha, hash_file, utc_now_iso, write_manifest


@dataclass
class GenericBuildConfig:
    config_path: Path
    payload: dict[str, Any]


def _load_config(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must deserialize to a mapping.")
    return data


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_payload(df: pd.DataFrame, date_col: str, target_col: str, split_cfg: dict[str, str]) -> dict[str, Any]:
    d = pd.to_datetime(df[date_col]).dt.normalize()
    train_end = pd.Timestamp(split_cfg["train_end"]).normalize()
    val_start = pd.Timestamp(split_cfg["val_start"]).normalize()
    val_end = pd.Timestamp(split_cfg["val_end"]).normalize()
    test_start = pd.Timestamp(split_cfg["test_start"]).normalize()
    masks = {
        "train": d <= train_end,
        "val": (d >= val_start) & (d <= val_end),
        "test": d >= test_start,
    }
    return {
        "generated_at_utc": utc_now_iso(),
        "date_column": date_col,
        "target_column": target_col,
        "boundaries": split_cfg,
        "counts": {k: int(v.sum()) for k, v in masks.items()},
        "dates": {k: d.loc[v].dt.strftime("%Y-%m-%d").tolist() for k, v in masks.items()},
    }


def build_generic_model_dataset(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    target_cfg = cfg["target"]
    feat_cfg = cfg["features"]
    split_cfg = cfg["splits"]
    out_cfg = cfg["outputs"]

    target_df = pd.read_parquet(target_cfg["path"])
    target_df = normalize_dates(target_df, frame_name="target_df", date_column=target_cfg["date_column"], require_unique=True)

    ar_df, ar_cols = build_target_features(
        target_df,
        date_col=target_cfg["date_column"],
        target_col=target_cfg["value_column"],
        lags=feat_cfg["target_lags"],
        rolling_windows=feat_cfg["target_rolling_windows"],
        momentum_pairs=[tuple(x) for x in feat_cfg.get("momentum_pairs", [])],
    )

    merged = ar_df.copy()
    exog_cols: list[str] = []
    for table in feat_cfg.get("exogenous_tables", []):
        exog_df = pd.read_parquet(table["path"])
        trailing_df, cols = build_trailing_exogenous_features(
            merged[target_cfg["date_column"]],
            exog_df,
            target_date_col=target_cfg["date_column"],
            exog_date_col=table["date_column"],
            exog_feature_columns=table["feature_columns"],
            windows=table["windows"],
            aggregations=table["aggregations"],
            lag_days=int(table.get("lag_days", feat_cfg.get("communication_lag_days", 0))),
            include_missing_indicators=bool(table.get("missingness_indicators", True)),
        )
        exog_cols.extend(cols)
        merged = merged.merge(trailing_df, on=target_cfg["date_column"], how="left")

    split_payload = _split_payload(merged, target_cfg["date_column"], target_cfg["value_column"], split_cfg)
    out_dataset = Path(out_cfg["dataset_path"])
    out_split = Path(out_cfg["split_path"])
    out_dataset.parent.mkdir(parents=True, exist_ok=True)
    out_split.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dataset, index=False)
    out_split.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    dataset_version = cfg.get("dataset_version") or default_dataset_version(prefix="model-dataset-t5yie-generic")
    manifest = ModelDatasetManifest(
        dataset_version=dataset_version,
        created_at_utc=utc_now_iso(),
        git_sha=get_git_sha(Path(__file__).resolve().parents[2]),
        target_column=target_cfg["value_column"],
        monthly_feature_columns=list(ar_cols + exog_cols),
        features_input_path=feat_cfg.get("features_input_path", "config-driven"),
        features_input_sha256=None,
        features_input_rows=0,
        target_input_path=str(target_cfg["path"]),
        target_input_sha256=hash_file(Path(target_cfg["path"])),
        target_input_rows=int(len(target_df)),
        output_dataset_path=str(out_dataset),
        output_dataset_sha256=hash_file(out_dataset),
        output_dataset_rows=int(len(merged)),
        split_output_path=str(out_split),
        split_output_sha256=hash_file(out_split),
        summary_output_path=None,
        summary_output_sha256=None,
        train_end=split_cfg["train_end"],
        val_start=split_cfg["val_start"],
        val_end=split_cfg["val_end"],
        test_start=split_cfg["test_start"],
    )
    write_manifest(
        out_dir=Path(out_cfg["manifest_out_dir"]),
        registry_path=Path(out_cfg["registry_path"]),
        manifest=manifest,
        extra={
            "builder_name": "generic_builder",
            "builder_version": "0.1.0",
            "config_path": str(config_path),
            "config_sha256": _sha256_text(config_path),
            "target_frequency": target_cfg.get("frequency", "D"),
            "forecast_horizon": target_cfg.get("forecast_horizon", 1),
            "target_lags": feat_cfg["target_lags"],
            "target_rolling_windows": feat_cfg["target_rolling_windows"],
            "exogenous_windows": [t["windows"] for t in feat_cfg.get("exogenous_tables", [])],
            "aggregation_functions": [t["aggregations"] for t in feat_cfg.get("exogenous_tables", [])],
            "communication_lag_days": feat_cfg.get("communication_lag_days", 0),
            "cutoff_policy": cfg.get("cutoff_policy", "target_date_minus_lag_days"),
            "no_lookahead_policy": cfg.get("no_lookahead_policy", "strict_trailing_windows"),
            "output_feature_columns": ar_cols + exog_cols,
            "dataset_shape": list(merged.shape),
            "missingness_summary": {c: float(merged[c].isna().mean()) for c in (ar_cols + exog_cols)},
        },
    )
    return out_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    out = build_generic_model_dataset(Path(args.config))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
