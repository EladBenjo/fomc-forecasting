"""Daily XGBoost runner for T5YIE diff forecasting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.evaluation.metrics import compute_metrics
from models.tracking.run_logger import (
    default_run_version,
    get_git_sha,
    hash_file,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMN = "t5yie_diff1"
DATE_COLUMN = "date"
MODEL_VARIANT = "xgboost_daily_all_features"

TARGET_FEATURE_COLUMNS: tuple[str, ...] = (
    "t5yie_diff1_lag_1",
    "t5yie_diff1_lag_2",
    "t5yie_diff1_lag_5",
    "t5yie_diff1_lag_10",
    "t5yie_diff1_roll_mean_5",
    "t5yie_diff1_roll_std_5",
    "t5yie_diff1_roll_mean_10",
    "t5yie_diff1_roll_std_10",
)

DAILY_COMM_WINDOWS: tuple[int, ...] = (7, 14, 30)


def _daily_event_columns(window_days: int) -> list[str]:
    return [
        f"hawkish_score_mean_{window_days}d",
        f"hawkish_score_sum_{window_days}d",
        f"hawkish_score_max_abs_signed_{window_days}d",
        f"novelty_mean_{window_days}d",
        f"n_target_sentences_sum_{window_days}d",
        f"doc_count_{window_days}d",
    ]


COMMUNICATION_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for window_days in DAILY_COMM_WINDOWS for col in _daily_event_columns(window_days)
)
FEATURE_COLUMNS: tuple[str, ...] = TARGET_FEATURE_COLUMNS + COMMUNICATION_FEATURE_COLUMNS

DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "targets" / "model_dataset_t5yie_daily.parquet"
DEFAULT_SPLITS_PATH = REPO_ROOT / "data" / "splits" / "time_splits_daily.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "models" / "xgboost" / "t5yie"


@dataclass
class XGBoostDailyConfig:
    """Configuration for deterministic daily XGBoost runs."""

    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    split_path: Path = field(default_factory=lambda: DEFAULT_SPLITS_PATH)
    output_root: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT)

    target_column: str = TARGET_COLUMN
    run_version: str | None = None
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42
    n_jobs: int = 1

    write_manifest_files: bool = True


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_write_json(payload: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp_path, out_path)


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def _load_xgb_regressor() -> type:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
        raise ImportError(
            "xgboost is required for the daily XGBoost runner. "
            "Install project dependencies with: pip install -e ."
        ) from exc
    return XGBRegressor


def _model_params(config: XGBoostDailyConfig) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": int(config.n_estimators),
        "max_depth": int(config.max_depth),
        "learning_rate": float(config.learning_rate),
        "subsample": float(config.subsample),
        "colsample_bytree": float(config.colsample_bytree),
        "random_state": int(config.random_state),
        "n_jobs": int(config.n_jobs),
        "tree_method": "hist",
        "verbosity": 0,
    }


def _load_boundaries(split_path: Path) -> dict[str, pd.Timestamp | None]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    boundaries = payload.get("boundaries", {})
    required = {"train_end", "val_start", "val_end", "test_start"}
    missing = required.difference(boundaries.keys())
    if missing:
        raise ValueError(f"Split artifact missing boundaries: {sorted(missing)}")

    return {
        "train_end": pd.Timestamp(boundaries["train_end"]).normalize(),
        "val_start": pd.Timestamp(boundaries["val_start"]).normalize(),
        "val_end": pd.Timestamp(boundaries["val_end"]).normalize(),
        "test_start": pd.Timestamp(boundaries["test_start"]).normalize(),
        "test_end": pd.Timestamp(boundaries["test_end"]).normalize()
        if boundaries.get("test_end")
        else None,
    }


def _load_model_frame(dataset_path: Path, target_column: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    required = {DATE_COLUMN, target_column, *FEATURE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Daily model dataset missing required columns for XGBoost: "
            f"{sorted(missing)}. Rebuild dataset with: python -m datasets.build_dataset.daily_builder"
        )

    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out = out.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    if not out[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Model dataset date index must be monotonic increasing.")
    if not out[DATE_COLUMN].is_unique:
        raise ValueError("Model dataset date index must be unique.")

    non_numeric = [
        col
        for col in (target_column, *FEATURE_COLUMNS)
        if not pd.api.types.is_numeric_dtype(out[col])
    ]
    if non_numeric:
        raise ValueError(f"XGBoost columns must be numeric: {non_numeric}")

    return out


def _split_masks(
    df: pd.DataFrame,
    boundaries: dict[str, pd.Timestamp | None],
) -> dict[str, pd.Series]:
    dates = df[DATE_COLUMN]
    train = dates <= boundaries["train_end"]
    val = (dates >= boundaries["val_start"]) & (dates <= boundaries["val_end"])
    test = dates >= boundaries["test_start"]
    if boundaries["test_end"] is not None:
        test = test & (dates <= boundaries["test_end"])
    return {"train": train, "val": val, "test": test}


def _fit_model(
    *,
    model_cls: type,
    params: dict[str, Any],
    df: pd.DataFrame,
    train_mask: pd.Series,
    target_column: str,
) -> Any:
    train = df.loc[train_mask & df[target_column].notna()].copy()
    if train.empty:
        raise ValueError("No training rows available for XGBoost.")
    model = model_cls(**params)
    model.fit(train[list(FEATURE_COLUMNS)], train[target_column])
    return model


def _predict_split(
    *,
    model: Any,
    df: pd.DataFrame,
    mask: pd.Series,
    target_column: str,
) -> pd.DataFrame:
    eval_df = df.loc[mask & df[target_column].notna()].copy()
    if eval_df.empty:
        return pd.DataFrame(columns=["date", "actual", "pred"])
    pred = model.predict(eval_df[list(FEATURE_COLUMNS)])
    out = pd.DataFrame(
        {
            "date": eval_df[DATE_COLUMN].to_numpy(),
            "actual": eval_df[target_column].astype(float).to_numpy(),
            "pred": np.asarray(pred, dtype=float),
        }
    )
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _build_results(predictions_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ["val", "test"]:
        pred_df = predictions_by_split[split_name]
        metrics = compute_metrics(pred_df, "actual", "pred")
        rows.append(
            {
                "model_variant": MODEL_VARIANT,
                "split": split_name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "directional_accuracy": metrics["directional_accuracy"],
                "n_forecasts": metrics["n_forecasts"],
                "feature_count": len(FEATURE_COLUMNS),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "model_variant"]).reset_index(drop=True)


def _build_summary(
    results_table: pd.DataFrame,
    *,
    target_column: str,
    train_rows: int,
    deployed_rows: int,
) -> dict[str, Any]:
    by_split = {str(row["split"]): row for row in results_table.to_dict(orient="records")}
    val = by_split.get("val", {})
    test = by_split.get("test", {})
    return {
        "model_variant": MODEL_VARIANT,
        "target_column": target_column,
        "feature_count": len(FEATURE_COLUMNS),
        "train_rows": int(train_rows),
        "deployed_train_rows": int(deployed_rows),
        "val_rmse": val.get("rmse"),
        "val_mae": val.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_mae": test.get("mae"),
        "test_directional_accuracy": test.get("directional_accuracy"),
        "recommendation": "Use XGBoost as a deployed daily artifact and compare test metrics against SARIMAX baselines.",
    }


def _feature_importance(model: Any) -> list[dict[str, float | str]]:
    raw = getattr(model, "feature_importances_", None)
    if raw is not None and len(raw) == len(FEATURE_COLUMNS):
        values = np.asarray(raw, dtype=float)
        return [
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(FEATURE_COLUMNS, values)
        ]

    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None:
        score = booster.get_score(importance_type="gain")
        return [
            {"feature": feature, "importance": float(score.get(feature, 0.0))}
            for feature in FEATURE_COLUMNS
        ]

    return [{"feature": feature, "importance": 0.0} for feature in FEATURE_COLUMNS]


def _feature_schema(df: pd.DataFrame, *, target_column: str) -> dict[str, Any]:
    return {
        "date_column": DATE_COLUMN,
        "target_column": target_column,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_dtypes": {col: str(df[col].dtype) for col in FEATURE_COLUMNS},
        "missing_rate": {col: float(df[col].isna().mean()) for col in FEATURE_COLUMNS},
    }


def _write_manifest(
    *,
    output_root: Path,
    run_version: str,
    payload: dict[str, Any],
) -> Path:
    manifest_path = output_root.parent / "manifests" / f"{run_version}.json"
    _atomic_write_json(payload, manifest_path)
    return manifest_path


def run_xgboost_daily(config: XGBoostDailyConfig) -> Path:
    """Train/evaluate the daily XGBoost model and write versioned artifacts."""
    dataset_path = _as_path(config.dataset_path)
    split_path = _as_path(config.split_path)
    output_root = _as_path(config.output_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset parquet: {dataset_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split artifact json: {split_path}")

    run_version = config.run_version or default_run_version(prefix="xgboost-t5yie-daily")
    output_dir = output_root / run_version
    if output_dir.exists():
        raise FileExistsError(f"Run output path already exists: {output_dir}")

    params = _model_params(config)
    model_df = _load_model_frame(dataset_path, config.target_column)
    boundaries = _load_boundaries(split_path)
    masks = _split_masks(model_df, boundaries)
    model_cls = _load_xgb_regressor()

    val_model = _fit_model(
        model_cls=model_cls,
        params=params,
        df=model_df,
        train_mask=masks["train"],
        target_column=config.target_column,
    )
    test_model = _fit_model(
        model_cls=model_cls,
        params=params,
        df=model_df,
        train_mask=masks["train"] | masks["val"],
        target_column=config.target_column,
    )
    deployed_model = _fit_model(
        model_cls=model_cls,
        params=params,
        df=model_df,
        train_mask=pd.Series(True, index=model_df.index),
        target_column=config.target_column,
    )

    val_pred = _predict_split(
        model=val_model,
        df=model_df,
        mask=masks["val"],
        target_column=config.target_column,
    )
    test_pred = _predict_split(
        model=test_model,
        df=model_df,
        mask=masks["test"],
        target_column=config.target_column,
    )
    predictions_by_split = {"val": val_pred, "test": test_pred}
    results_table = _build_results(predictions_by_split)

    prediction_frames: list[pd.DataFrame] = []
    for split_name, pred_df in predictions_by_split.items():
        if pred_df.empty:
            continue
        framed = pred_df.copy()
        framed.insert(0, "split", split_name)
        framed.insert(0, "model_variant", MODEL_VARIANT)
        framed.insert(0, "run_version", run_version)
        prediction_frames.append(framed[["run_version", "model_variant", "split", "date", "actual", "pred"]])

    if prediction_frames:
        predictions_df = (
            pd.concat(prediction_frames, ignore_index=True)
            .sort_values(["split", "model_variant", "date"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        predictions_df = pd.DataFrame(
            columns=["run_version", "model_variant", "split", "date", "actual", "pred"]
        )

    train_rows = int((masks["train"] & model_df[config.target_column].notna()).sum())
    deployed_rows = int(model_df[config.target_column].notna().sum())
    run_summary = _build_summary(
        results_table,
        target_column=config.target_column,
        train_rows=train_rows,
        deployed_rows=deployed_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.parquet"
    results_path = output_dir / "results_table.json"
    summary_path = output_dir / "run_summary.json"
    config_path = output_dir / "run_config.json"
    importance_path = output_dir / "feature_importance.json"
    model_path = output_dir / "model.json"
    schema_path = output_dir / "feature_schema.json"

    _atomic_write_parquet(predictions_df, predictions_path)
    _atomic_write_json(results_table.to_dict(orient="records"), results_path)
    _atomic_write_json(run_summary, summary_path)
    _atomic_write_json(_feature_importance(deployed_model), importance_path)
    _atomic_write_json(_feature_schema(model_df, target_column=config.target_column), schema_path)
    deployed_model.save_model(model_path)
    _atomic_write_json(
        {
            "run_version": run_version,
            "dataset_path": str(dataset_path),
            "split_path": str(split_path),
            "target_column": config.target_column,
            "model_variant": MODEL_VARIANT,
            "feature_columns": list(FEATURE_COLUMNS),
            "xgboost_params": params,
            "split_boundaries": {
                key: value.strftime("%Y-%m-%d") if value is not None else None
                for key, value in boundaries.items()
            },
        },
        config_path,
    )

    if config.write_manifest_files:
        manifest_payload = {
            "run_version": run_version,
            "created_at_utc": utc_now_iso(),
            "git_sha": get_git_sha(REPO_ROOT),
            "model_family": "xgboost",
            "model_variant": MODEL_VARIANT,
            "target_column": config.target_column,
            "feature_columns": list(FEATURE_COLUMNS),
            "xgboost_params": params,
            "dataset_input_path": str(dataset_path),
            "dataset_input_sha256": hash_file(dataset_path),
            "dataset_input_rows": len(model_df),
            "splits_input_path": str(split_path),
            "splits_input_sha256": hash_file(split_path),
            "output_dir": str(output_dir),
            "outputs": {
                "predictions": {
                    "path": str(predictions_path),
                    "sha256": hash_file(predictions_path),
                    "rows": len(predictions_df),
                },
                "results_table": {"path": str(results_path), "sha256": hash_file(results_path)},
                "run_summary": {"path": str(summary_path), "sha256": hash_file(summary_path)},
                "run_config": {"path": str(config_path), "sha256": hash_file(config_path)},
                "feature_importance": {
                    "path": str(importance_path),
                    "sha256": hash_file(importance_path),
                },
                "feature_schema": {"path": str(schema_path), "sha256": hash_file(schema_path)},
                "model": {"path": str(model_path), "sha256": hash_file(model_path)},
            },
            "split_boundaries": {
                key: value.strftime("%Y-%m-%d") if value is not None else None
                for key, value in boundaries.items()
            },
            "train_rows": train_rows,
            "val_rows": int((masks["val"] & model_df[config.target_column].notna()).sum()),
            "test_rows": int((masks["test"] & model_df[config.target_column].notna()).sum()),
            "deployed_train_rows": deployed_rows,
        }
        manifest_path = _write_manifest(
            output_root=output_root,
            run_version=run_version,
            payload=manifest_payload,
        )
        print(f"wrote manifest: {manifest_path}")

    print(f"wrote predictions: {predictions_path}")
    print(f"wrote results: {results_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote config: {config_path}")
    print(f"wrote feature importance: {importance_path}")
    print(f"wrote feature schema: {schema_path}")
    print(f"wrote model: {model_path}")
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run daily T5YIE XGBoost training/evaluation.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to model dataset parquet (default: data/targets/model_dataset_t5yie_daily.parquet).",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to split artifact json (default: data/splits/time_splits_daily.json).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory for versioned run artifacts (default: data/models/xgboost/t5yie).",
    )
    parser.add_argument(
        "--run-version",
        type=str,
        default=None,
        help="Optional explicit run version (default: autogenerated UTC tag).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=TARGET_COLUMN,
        help=f"Target column name in model dataset (default: {TARGET_COLUMN}).",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-manifest",
        dest="write_manifest_files",
        action="store_false",
        default=True,
        help="Disable manifest JSON write.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> XGBoostDailyConfig:
    config = XGBoostDailyConfig(
        target_column=args.target_column,
        run_version=args.run_version,
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.random_state),
        write_manifest_files=bool(args.write_manifest_files),
    )
    if args.dataset_path:
        config.dataset_path = Path(args.dataset_path)
    if args.split_path:
        config.split_path = Path(args.split_path)
    if args.output_root:
        config.output_root = Path(args.output_root)
    return config


def main(argv: list[str] | None = None) -> Path:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    output_dir = run_xgboost_daily(config)
    print(f"xgboost run dir: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = ["XGBoostDailyConfig", "run_xgboost_daily", "main"]
