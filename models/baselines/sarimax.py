"""Phase 4 benchmark runner: baseline vs exogenous SARIMAX variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.evaluation.metrics import compute_metrics
from models.evaluation.walk_forward import run_expanding_one_step_sarimax
from models.tracking.run_logger import (
    BaselineRunManifest,
    default_run_version,
    get_git_sha,
    hash_file,
    utc_now_iso,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMN = "t5yie_diff1"
DATE_COLUMN = "date"

MODEL_VARIANTS: dict[str, list[str]] = {
    "baseline_univariate": [],
    "exog_minimal_counts": ["hawkish_score", "novelty", "doc_count"],
    "exog_share_variant": ["hawkish_score", "novelty", "hawkish_share", "dovish_share"],
}

DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "targets" / "model_dataset_t5yie.parquet"
DEFAULT_SPLITS_PATH = REPO_ROOT / "data" / "splits" / "time_splits.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "models" / "baselines" / "t5yie"


@dataclass
class SarimaxBenchmarkConfig:
    """Configuration for deterministic Phase 4 SARIMAX benchmark runs."""

    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    split_path: Path = field(default_factory=lambda: DEFAULT_SPLITS_PATH)
    output_root: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT)

    target_column: str = TARGET_COLUMN
    run_version: str | None = None
    model_order: tuple[int, int, int] = (1, 0, 0)
    model_trend: str = "c"
    min_train_obs: int = 36
    meaningful_threshold_pct: float = 2.0

    write_manifest_files: bool = True
    manifest_out_dir: Path | None = None
    manifest_registry_path: Path | None = None


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


def _load_benchmark_frame(dataset_path: Path, target_column: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    required = {
        DATE_COLUMN,
        target_column,
        "hawkish_score",
        "novelty",
        "n_hawkish",
        "n_dovish",
        "n_target_sentences",
        "doc_count",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Model dataset missing required columns: {sorted(missing)}")

    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out = out.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    if not out[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Model dataset date index must be monotonic increasing.")
    if not out[DATE_COLUMN].is_unique:
        raise ValueError("Model dataset date index must be unique.")

    out["hawkish_share"] = np.where(
        out["n_target_sentences"] > 0,
        out["n_hawkish"] / out["n_target_sentences"],
        np.nan,
    )
    out["dovish_share"] = np.where(
        out["n_target_sentences"] > 0,
        out["n_dovish"] / out["n_target_sentences"],
        np.nan,
    )
    return out


def _build_paired_comparison(
    all_predictions: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    paired_rows: list[dict[str, float | int | str]] = []
    for exog_variant in ["exog_minimal_counts", "exog_share_variant"]:
        for split_name in ["val", "test"]:
            base_pred = all_predictions[("baseline_univariate", split_name)]
            exog_pred = all_predictions[(exog_variant, split_name)]

            common_dates = sorted(set(base_pred["date"]) & set(exog_pred["date"]))
            if not common_dates:
                paired_rows.append(
                    {
                        "split": split_name,
                        "exog_variant": exog_variant,
                        "n_common_dates": 0,
                        "baseline_rmse": float("nan"),
                        "exog_rmse": float("nan"),
                        "rmse_improvement_pct": float("nan"),
                        "baseline_mae": float("nan"),
                        "exog_mae": float("nan"),
                        "mae_improvement_pct": float("nan"),
                    }
                )
                continue

            b = base_pred[base_pred["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
            e = exog_pred[exog_pred["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
            if not bool((b["date"].values == e["date"].values).all()):
                raise ValueError("Timestamp mismatch in paired comparison.")

            b_metrics = compute_metrics(b, "actual", "pred")
            e_metrics = compute_metrics(e, "actual", "pred")

            b_rmse = float(b_metrics["rmse"])
            e_rmse = float(e_metrics["rmse"])
            b_mae = float(b_metrics["mae"])
            e_mae = float(e_metrics["mae"])

            paired_rows.append(
                {
                    "split": split_name,
                    "exog_variant": exog_variant,
                    "n_common_dates": len(common_dates),
                    "baseline_rmse": b_rmse,
                    "exog_rmse": e_rmse,
                    "rmse_improvement_pct": 100.0 * (b_rmse - e_rmse) / b_rmse if b_rmse else float("nan"),
                    "baseline_mae": b_mae,
                    "exog_mae": e_mae,
                    "mae_improvement_pct": 100.0 * (b_mae - e_mae) / b_mae if b_mae else float("nan"),
                }
            )
    return pd.DataFrame(paired_rows).sort_values(["split", "exog_variant"]).reset_index(drop=True)


def _build_run_summary(
    *,
    paired_comparison: pd.DataFrame,
    meaningful_threshold_pct: float,
) -> dict[str, Any]:
    paired_test = paired_comparison[paired_comparison["split"] == "test"].copy()
    if paired_test.empty:
        return {
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "best_exog_variant": None,
            "rmse_gain_pct": float("nan"),
            "mae_gain_pct": float("nan"),
            "meaningful_threshold_pct": meaningful_threshold_pct,
            "is_meaningful": False,
            "recommendation": "No test paired comparison available.",
        }

    best_paired = paired_test.sort_values(
        ["rmse_improvement_pct", "mae_improvement_pct"],
        ascending=False,
        na_position="last",
    ).iloc[0]
    rmse_gain = float(best_paired["rmse_improvement_pct"])
    mae_gain = float(best_paired["mae_improvement_pct"])
    best_variant = str(best_paired["exog_variant"])
    is_meaningful = (rmse_gain >= meaningful_threshold_pct) and (mae_gain >= meaningful_threshold_pct)

    if is_meaningful:
        recommendation = (
            f"Use univariate baseline + {best_variant} as primary exogenous benchmark "
            f"(RMSE +{rmse_gain:.2f}%, MAE +{mae_gain:.2f}%)."
        )
    else:
        recommendation = (
            "Keep univariate baseline as default forecasting path; treat exogenous features as "
            f"diagnostic until lift exceeds {meaningful_threshold_pct:.1f}% on both RMSE and MAE."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "best_exog_variant": best_variant,
        "rmse_gain_pct": rmse_gain,
        "mae_gain_pct": mae_gain,
        "meaningful_threshold_pct": meaningful_threshold_pct,
        "is_meaningful": is_meaningful,
        "recommendation": recommendation,
    }


def run_phase4_benchmark(config: SarimaxBenchmarkConfig) -> Path:
    """Run Phase 4 SARIMAX benchmark and write versioned run artifacts."""
    dataset_path = _as_path(config.dataset_path)
    split_path = _as_path(config.split_path)
    output_root = _as_path(config.output_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset parquet: {dataset_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split artifact json: {split_path}")

    run_version = config.run_version or default_run_version()
    output_dir = output_root / run_version
    if output_dir.exists():
        raise FileExistsError(f"Run output path already exists: {output_dir}")

    benchmark_df = _load_benchmark_frame(dataset_path, config.target_column)
    boundaries = _load_boundaries(split_path)

    splits = {
        "val": (boundaries["val_start"], boundaries["val_end"]),
        "test": (boundaries["test_start"], boundaries["test_end"]),
    }

    all_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    for variant_name, exog_cols in MODEL_VARIANTS.items():
        for split_name in ["val", "test"]:
            split_start, split_end = splits[split_name]
            pred_df, n_fail = run_expanding_one_step_sarimax(
                df=benchmark_df,
                target_col=config.target_column,
                date_col=DATE_COLUMN,
                eval_start=split_start,
                eval_end=split_end,
                exog_cols=exog_cols,
                order=config.model_order,
                trend=config.model_trend,
                min_train_obs=config.min_train_obs,
            )

            if not pred_df.empty:
                if not pred_df["date"].is_monotonic_increasing:
                    raise ValueError("Prediction dates must be monotonic increasing.")
                if split_start is not None and not bool((pred_df["date"] >= split_start).all()):
                    raise ValueError("Prediction row fell before split start.")
                if split_end is not None and not bool((pred_df["date"] <= split_end).all()):
                    raise ValueError("Prediction row fell after split end.")

            metrics = compute_metrics(pred_df, "actual", "pred")
            rows.append(
                {
                    "model_variant": variant_name,
                    "split": split_name,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "directional_accuracy": metrics["directional_accuracy"],
                    "n_forecasts": metrics["n_forecasts"],
                    "n_fit_failures": int(n_fail),
                    "exog_cols": ", ".join(exog_cols) if exog_cols else "[none]",
                }
            )
            all_predictions[(variant_name, split_name)] = pred_df

    results_table = pd.DataFrame(rows).sort_values(["split", "rmse", "mae"]).reset_index(drop=True)
    paired_comparison = _build_paired_comparison(all_predictions)
    run_summary = _build_run_summary(
        paired_comparison=paired_comparison,
        meaningful_threshold_pct=config.meaningful_threshold_pct,
    )

    prediction_frames: list[pd.DataFrame] = []
    for (variant, split_name), pred_df in all_predictions.items():
        if pred_df.empty:
            continue
        framed = pred_df.copy()
        framed.insert(0, "split", split_name)
        framed.insert(0, "model_variant", variant)
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

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.parquet"
    results_path = output_dir / "results_table.json"
    paired_path = output_dir / "paired_comparison.json"
    summary_path = output_dir / "run_summary.json"
    config_path = output_dir / "run_config.json"

    _atomic_write_parquet(predictions_df, predictions_path)
    _atomic_write_json(results_table.to_dict(orient="records"), results_path)
    _atomic_write_json(paired_comparison.to_dict(orient="records"), paired_path)
    _atomic_write_json(run_summary, summary_path)
    _atomic_write_json(
        {
            "run_version": run_version,
            "dataset_path": str(dataset_path),
            "split_path": str(split_path),
            "target_column": config.target_column,
            "model_order": list(config.model_order),
            "model_trend": config.model_trend,
            "min_train_obs": config.min_train_obs,
            "meaningful_threshold_pct": config.meaningful_threshold_pct,
            "model_variants": MODEL_VARIANTS,
            "split_boundaries": {
                key: value.strftime("%Y-%m-%d") if value is not None else None
                for key, value in boundaries.items()
            },
        },
        config_path,
    )

    if config.write_manifest_files:
        manifest_out_dir = _as_path(config.manifest_out_dir) if config.manifest_out_dir else output_root.parent
        manifest_registry_path = (
            _as_path(config.manifest_registry_path)
            if config.manifest_registry_path
            else manifest_out_dir / "run_registry.sqlite3"
        )
        manifest = BaselineRunManifest(
            run_version=run_version,
            created_at_utc=utc_now_iso(),
            git_sha=get_git_sha(REPO_ROOT),
            target_column=config.target_column,
            model_order=list(config.model_order),
            model_trend=config.model_trend,
            min_train_obs=config.min_train_obs,
            meaningful_threshold_pct=config.meaningful_threshold_pct,
            model_variants_json=json.dumps(MODEL_VARIANTS, sort_keys=True),
            dataset_input_path=str(dataset_path),
            dataset_input_sha256=hash_file(dataset_path),
            dataset_input_rows=len(benchmark_df),
            splits_input_path=str(split_path),
            splits_input_sha256=hash_file(split_path),
            output_dir=str(output_dir),
            predictions_output_path=str(predictions_path),
            predictions_output_sha256=hash_file(predictions_path),
            predictions_output_rows=len(predictions_df),
            results_output_path=str(results_path),
            results_output_sha256=hash_file(results_path),
            paired_output_path=str(paired_path),
            paired_output_sha256=hash_file(paired_path),
            summary_output_path=str(summary_path),
            summary_output_sha256=hash_file(summary_path),
            config_output_path=str(config_path),
            config_output_sha256=hash_file(config_path),
            val_start=boundaries["val_start"].strftime("%Y-%m-%d"),
            val_end=boundaries["val_end"].strftime("%Y-%m-%d"),
            test_start=boundaries["test_start"].strftime("%Y-%m-%d"),
            test_end=boundaries["test_end"].strftime("%Y-%m-%d")
            if boundaries["test_end"] is not None
            else None,
        )
        manifest_path = write_manifest(
            out_dir=manifest_out_dir,
            registry_path=manifest_registry_path,
            manifest=manifest,
            extra={
                "best_exog_variant": run_summary.get("best_exog_variant"),
                "is_meaningful": run_summary.get("is_meaningful"),
            },
        )
        print(f"wrote manifest: {manifest_path}")

    print(f"wrote predictions: {predictions_path}")
    print(f"wrote results: {results_path}")
    print(f"wrote paired comparison: {paired_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote config: {config_path}")
    return output_dir


def _parse_model_order(raw: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--model-order must include exactly three comma-separated ints (p,d,q).")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--model-order must be int,int,int.") from exc
    return order  # type: ignore[return-value]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4 baseline-vs-exogenous SARIMAX benchmark.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to model dataset parquet (default: data/targets/model_dataset_t5yie.parquet).",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to split artifact json (default: data/splits/time_splits.json).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory for versioned run artifacts (default: data/models/baselines/t5yie).",
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
    parser.add_argument(
        "--model-order",
        type=str,
        default="1,0,0",
        help="SARIMAX order as comma-separated p,d,q (default: 1,0,0).",
    )
    parser.add_argument(
        "--model-trend",
        type=str,
        default="c",
        help="SARIMAX trend parameter (default: c).",
    )
    parser.add_argument(
        "--min-train-obs",
        type=int,
        default=36,
        help="Minimum train observations per fold (default: 36).",
    )
    parser.add_argument(
        "--meaningful-threshold-pct",
        type=float,
        default=2.0,
        help="Threshold for meaningful test lift on both RMSE and MAE (default: 2.0).",
    )
    parser.add_argument(
        "--manifest-out-dir",
        type=str,
        default=None,
        help="Optional directory for run manifest JSON output (default: output root parent).",
    )
    parser.add_argument(
        "--manifest-registry-path",
        type=str,
        default=None,
        help="Optional path for run registry sqlite (default: <manifest_out_dir>/run_registry.sqlite3).",
    )
    parser.add_argument(
        "--manifest",
        dest="write_manifest_files",
        action="store_true",
        default=True,
        help="Write run manifest + registry row (default: enabled).",
    )
    parser.add_argument(
        "--no-manifest",
        dest="write_manifest_files",
        action="store_false",
        help="Disable manifest/registry writes.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SarimaxBenchmarkConfig:
    config = SarimaxBenchmarkConfig(
        target_column=args.target_column,
        run_version=args.run_version,
        model_order=_parse_model_order(args.model_order),
        model_trend=args.model_trend,
        min_train_obs=int(args.min_train_obs),
        meaningful_threshold_pct=float(args.meaningful_threshold_pct),
        write_manifest_files=bool(args.write_manifest_files),
    )
    if args.dataset_path:
        config.dataset_path = Path(args.dataset_path)
    if args.split_path:
        config.split_path = Path(args.split_path)
    if args.output_root:
        config.output_root = Path(args.output_root)
    if args.manifest_out_dir:
        config.manifest_out_dir = Path(args.manifest_out_dir)
    if args.manifest_registry_path:
        config.manifest_registry_path = Path(args.manifest_registry_path)
    return config


def main(argv: list[str] | None = None) -> Path:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    output_dir = run_phase4_benchmark(config)
    print(f"phase4 run dir: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = ["SarimaxBenchmarkConfig", "run_phase4_benchmark", "main"]

