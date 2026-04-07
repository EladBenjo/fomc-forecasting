from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models.evaluation.metrics import compute_metrics, directional_accuracy, mae, rmse


def test_rmse_mae_directional_accuracy_values():
    y_true = pd.Series([1.0, -2.0, 3.0, -4.0])
    y_pred = pd.Series([0.0, -1.0, 2.0, -2.0])

    assert math.isclose(rmse(y_true, y_pred), math.sqrt(1.75), rel_tol=1e-9)
    assert math.isclose(mae(y_true, y_pred), 1.25, rel_tol=1e-9)
    assert math.isclose(directional_accuracy(y_true, y_pred), 0.75, rel_tol=1e-9)


def test_compute_metrics_empty_after_dropna():
    frame = pd.DataFrame({"actual": [np.nan, np.nan], "pred": [1.0, np.nan]})
    metrics = compute_metrics(frame, y_col="actual", yhat_col="pred")

    assert math.isnan(float(metrics["rmse"]))
    assert math.isnan(float(metrics["mae"]))
    assert math.isnan(float(metrics["directional_accuracy"]))
    assert metrics["n_forecasts"] == 0


def test_compute_metrics_raises_on_missing_columns():
    frame = pd.DataFrame({"actual": [1.0]})
    try:
        compute_metrics(frame)
    except ValueError as exc:
        assert "Missing columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError on missing prediction column.")
