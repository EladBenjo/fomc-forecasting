"""Walk-forward evaluation helpers for SARIMAX benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
except ImportError:  # pragma: no cover - covered indirectly in CLI/runtime checks.
    _SARIMAX = None


def split_mask(date_series: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    """Build an inclusive split mask over a date column."""
    normalized = pd.to_datetime(date_series, errors="coerce").dt.normalize()
    if normalized.isna().any():
        raise ValueError("date_series contains invalid timestamps.")

    mask = pd.Series(True, index=date_series.index)
    if start is not None:
        mask = mask & (normalized >= pd.Timestamp(start).normalize())
    if end is not None:
        mask = mask & (normalized <= pd.Timestamp(end).normalize())
    return mask


def _prepare_fold_data(
    *,
    work: pd.DataFrame,
    idx: int,
    target_col: str,
    date_col: str,
    exog_cols: Sequence[str],
    min_train_obs: int,
) -> tuple[pd.Timestamp, pd.Series, pd.DataFrame | None, pd.DataFrame | None] | None:
    forecast_date = work.loc[idx, date_col]
    train = work.loc[work[date_col] < forecast_date].copy()
    if len(train) < min_train_obs:
        return None
    if not bool((train[date_col] < forecast_date).all()):
        raise ValueError("Leakage check failed: train fold includes forecast date/future.")

    train_y = train[target_col]

    if exog_cols:
        train_exog = train[list(exog_cols)]
        test_exog = work.loc[[idx], list(exog_cols)]
        if bool(test_exog.isna().any(axis=1).iloc[0]):
            return None

        valid_train = train_y.notna() & (~train_exog.isna().any(axis=1))
        train_y = train_y.loc[valid_train]
        train_exog = train_exog.loc[valid_train]
        if len(train_y) < min_train_obs:
            return None

        if bool(train_exog.isna().any().any()):
            raise ValueError("NaN exogenous values in training fold after filtering.")
        if bool(test_exog.isna().any().any()):
            raise ValueError("NaN exogenous values in forecast row.")
    else:
        train_exog = None
        test_exog = None
        train_y = train_y.dropna()
        if len(train_y) < min_train_obs:
            return None

    return forecast_date, train_y, train_exog, test_exog


def _run_refit_one_step_sarimax(
    *,
    work: pd.DataFrame,
    target_col: str,
    date_col: str,
    eval_idx: pd.Index,
    exog_cols: Sequence[str],
    order: tuple[int, int, int],
    trend: str,
    min_train_obs: int,
) -> tuple[pd.DataFrame, int]:
    predictions: list[dict[str, float | pd.Timestamp]] = []
    n_failures = 0

    for idx in eval_idx:
        fold_data = _prepare_fold_data(
            work=work,
            idx=int(idx),
            target_col=target_col,
            date_col=date_col,
            exog_cols=exog_cols,
            min_train_obs=min_train_obs,
        )
        if fold_data is None:
            continue

        forecast_date, train_y, train_exog, test_exog = fold_data

        try:
            model = _SARIMAX(
                endog=train_y,
                exog=train_exog,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_res = model.fit(disp=False)
            if test_exog is None:
                forecast = fit_res.forecast(steps=1)
            else:
                forecast = fit_res.forecast(steps=1, exog=test_exog)
            pred = float(pd.Series(forecast).iloc[0])
        except Exception:
            n_failures += 1
            continue

        predictions.append(
            {
                "date": forecast_date,
                "actual": float(work.loc[idx, target_col]),
                "pred": pred,
            }
        )

    pred_df = pd.DataFrame(predictions, columns=["date", "actual", "pred"])
    if not pred_df.empty:
        pred_df = pred_df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return pred_df, n_failures


def _run_append_one_step_sarimax(
    *,
    work: pd.DataFrame,
    target_col: str,
    date_col: str,
    eval_idx: pd.Index,
    exog_cols: Sequence[str],
    order: tuple[int, int, int],
    trend: str,
    min_train_obs: int,
) -> tuple[pd.DataFrame, int]:
    predictions: list[dict[str, float | pd.Timestamp]] = []
    n_failures = 0
    fit_res = None
    next_obs_index: int | None = None

    for idx in eval_idx:
        idx = int(idx)
        if fit_res is None:
            fold_data = _prepare_fold_data(
                work=work,
                idx=idx,
                target_col=target_col,
                date_col=date_col,
                exog_cols=exog_cols,
                min_train_obs=min_train_obs,
            )
            if fold_data is None:
                continue

            forecast_date, train_y, train_exog, test_exog = fold_data
            train_y = train_y.reset_index(drop=True)
            if train_exog is not None:
                train_exog = train_exog.reset_index(drop=True)

            try:
                model = _SARIMAX(
                    endog=train_y,
                    exog=train_exog,
                    order=order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_res = model.fit(disp=False)
            except Exception:
                n_failures += 1
                fit_res = None
                continue

            next_obs_index = len(train_y)
        else:
            forecast_date = work.loc[idx, date_col]
            if exog_cols:
                test_exog = work.loc[[idx], list(exog_cols)]
                if bool(test_exog.isna().any(axis=1).iloc[0]):
                    continue
            else:
                test_exog = None

        forecast_exog = None
        if test_exog is not None:
            if next_obs_index is None:
                raise ValueError("Internal SARIMAX append state missing next observation index.")
            forecast_exog = pd.DataFrame(
                test_exog.to_numpy(),
                columns=list(exog_cols),
                index=[next_obs_index],
            )

        try:
            if forecast_exog is None:
                forecast = fit_res.forecast(steps=1)
            else:
                forecast = fit_res.forecast(steps=1, exog=forecast_exog)
            pred = float(pd.Series(forecast).iloc[0])
        except Exception:
            n_failures += 1
            pred = float("nan")

        actual_value = work.loc[idx, target_col]
        predictions.append(
            {
                "date": forecast_date,
                "actual": float(actual_value),
                "pred": pred,
            }
        )

        if pd.isna(actual_value):
            continue
        if next_obs_index is None:
            raise ValueError("Internal SARIMAX append state missing next observation index.")

        append_y = pd.Series([float(actual_value)], index=[next_obs_index], name=target_col)
        append_exog = forecast_exog if exog_cols else None
        try:
            fit_res = fit_res.append(append_y, exog=append_exog, refit=False)
            next_obs_index += 1
        except Exception:
            n_failures += 1

    pred_df = pd.DataFrame(predictions, columns=["date", "actual", "pred"])
    if not pred_df.empty:
        pred_df = pred_df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return pred_df, n_failures


def run_expanding_one_step_sarimax(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    eval_start: pd.Timestamp | None,
    eval_end: pd.Timestamp | None,
    exog_cols: Sequence[str],
    order: tuple[int, int, int] = (1, 0, 0),
    trend: str = "c",
    min_train_obs: int = 36,
) -> tuple[pd.DataFrame, int]:
    """
    Run leakage-safe expanding-window one-step SARIMAX forecasting.

    Returns:
        - predictions DataFrame with columns [date, actual, pred]
        - fit failure count
    """
    if _SARIMAX is None:
        raise ImportError(
            "statsmodels is required for SARIMAX benchmarking. "
            "Install with: pip install statsmodels>=0.14"
        )
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col!r}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col!r}")

    missing_exog = [col for col in exog_cols if col not in df.columns]
    if missing_exog:
        raise ValueError(f"Missing exogenous columns: {missing_exog}")
    if min_train_obs <= 0:
        raise ValueError("min_train_obs must be positive.")

    work = df.sort_values(date_col, kind="mergesort").reset_index(drop=True).copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    if work[date_col].isna().any():
        raise ValueError("Found invalid timestamps in date column after normalization.")

    eval_mask = split_mask(work[date_col], eval_start, eval_end)
    eval_idx = work.index[eval_mask]

    if getattr(_SARIMAX, "__module__", "").startswith("statsmodels."):
        return _run_append_one_step_sarimax(
            work=work,
            target_col=target_col,
            date_col=date_col,
            eval_idx=eval_idx,
            exog_cols=exog_cols,
            order=order,
            trend=trend,
            min_train_obs=min_train_obs,
        )

    return _run_refit_one_step_sarimax(
        work=work,
        target_col=target_col,
        date_col=date_col,
        eval_idx=eval_idx,
        exog_cols=exog_cols,
        order=order,
        trend=trend,
        min_train_obs=min_train_obs,
    )


__all__ = ["split_mask", "run_expanding_one_step_sarimax"]
