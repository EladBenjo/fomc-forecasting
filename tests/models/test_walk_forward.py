from __future__ import annotations

import numpy as np
import pandas as pd

import models.evaluation.walk_forward as walk_forward


class _FitResult:
    def __init__(self, train_y: pd.Series):
        self._train_y = train_y

    def forecast(self, steps: int = 1, exog: pd.DataFrame | None = None) -> pd.Series:
        return pd.Series([float(self._train_y.iloc[-1])])


class _FakeSARIMAX:
    def __init__(
        self,
        *,
        endog: pd.Series,
        exog: pd.DataFrame | None,
        order: tuple[int, int, int],
        trend: str,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
    ):
        self._endog = endog

    def fit(self, disp: bool = False) -> _FitResult:
        return _FitResult(self._endog)


def test_run_expanding_one_step_sarimax_enforces_no_leakage(monkeypatch):
    monkeypatch.setattr(walk_forward, "_SARIMAX", _FakeSARIMAX)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    pred_df, n_fail = walk_forward.run_expanding_one_step_sarimax(
        df=df,
        target_col="y",
        date_col="date",
        eval_start=pd.Timestamp("2020-02-01"),
        eval_end=pd.Timestamp("2020-04-01"),
        exog_cols=[],
        min_train_obs=2,
    )

    assert n_fail == 0
    assert pred_df["date"].tolist() == [
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2020-04-01"),
    ]


def test_run_expanding_one_step_sarimax_skips_nan_exog_rows(monkeypatch):
    monkeypatch.setattr(walk_forward, "_SARIMAX", _FakeSARIMAX)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                    "2020-05-01",
                ]
            ),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [0.0, np.nan, 0.0, np.nan, 0.0],
        }
    )

    pred_df, n_fail = walk_forward.run_expanding_one_step_sarimax(
        df=df,
        target_col="y",
        date_col="date",
        eval_start=pd.Timestamp("2020-03-01"),
        eval_end=pd.Timestamp("2020-05-01"),
        exog_cols=["x"],
        min_train_obs=2,
    )

    assert n_fail == 0
    assert pred_df["date"].tolist() == [pd.Timestamp("2020-05-01")]


def test_run_expanding_one_step_sarimax_counts_fit_failures(monkeypatch):
    class _FailsFirstSARIMAX(_FakeSARIMAX):
        _calls = 0

        def fit(self, disp: bool = False) -> _FitResult:
            type(self)._calls += 1
            if type(self)._calls == 1:
                raise RuntimeError("synthetic fit failure")
            return super().fit(disp=disp)

    monkeypatch.setattr(walk_forward, "_SARIMAX", _FailsFirstSARIMAX)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                    "2020-05-01",
                    "2020-06-01",
                ]
            ),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    pred_df, n_fail = walk_forward.run_expanding_one_step_sarimax(
        df=df,
        target_col="y",
        date_col="date",
        eval_start=pd.Timestamp("2020-03-01"),
        eval_end=pd.Timestamp("2020-06-01"),
        exog_cols=[],
        min_train_obs=2,
    )

    assert n_fail == 1
    assert len(pred_df) == 3


def test_run_expanding_one_step_sarimax_outputs_monotonic_dates(monkeypatch):
    monkeypatch.setattr(walk_forward, "_SARIMAX", _FakeSARIMAX)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-01", "2020-01-01", "2020-03-01", "2020-02-01"]),
            "y": [4.0, 1.0, 3.0, 2.0],
        }
    )

    pred_df, _ = walk_forward.run_expanding_one_step_sarimax(
        df=df,
        target_col="y",
        date_col="date",
        eval_start=pd.Timestamp("2020-02-01"),
        eval_end=pd.Timestamp("2020-04-01"),
        exog_cols=[],
        min_train_obs=1,
    )

    assert pred_df["date"].is_monotonic_increasing

