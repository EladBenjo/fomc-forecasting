"""Metric helpers for Phase 4 baseline benchmarking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root mean squared error."""
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean absolute error."""
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def directional_accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Share of observations where sign(actual) equals sign(prediction)."""
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if y_true_arr.size == 0:
        return float("nan")
    return float((np.sign(y_true_arr) == np.sign(y_pred_arr)).mean())


def compute_metrics(df: pd.DataFrame, y_col: str = "actual", yhat_col: str = "pred") -> dict[str, float | int]:
    """Compute benchmark metrics with notebook-compatible empty-frame behavior."""
    required = {y_col, yhat_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for metric computation: {sorted(missing)}")

    work = df[[y_col, yhat_col]].dropna()
    if work.empty:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "n_forecasts": 0,
        }

    return {
        "rmse": rmse(work[y_col], work[yhat_col]),
        "mae": mae(work[y_col], work[yhat_col]),
        "directional_accuracy": directional_accuracy(work[y_col], work[yhat_col]),
        "n_forecasts": int(len(work)),
    }


__all__ = ["rmse", "mae", "directional_accuracy", "compute_metrics"]

