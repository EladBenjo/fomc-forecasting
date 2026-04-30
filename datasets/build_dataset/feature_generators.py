from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd


AGG_FUNCS = {"mean", "sum", "max", "last", "count"}


def build_target_features(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    momentum_pairs: Sequence[tuple[int, int]] = (),
) -> tuple[pd.DataFrame, list[str]]:
    out = df[[date_col, target_col]].copy()
    s = out[target_col]
    created: list[str] = []
    for lag in lags:
        col = f"{target_col}_lag_{lag}"
        out[col] = s.shift(int(lag))
        created.append(col)

    shifted = s.shift(1)
    for win in rolling_windows:
        mc = f"{target_col}_roll_mean_{win}"
        sc = f"{target_col}_roll_std_{win}"
        out[mc] = shifted.rolling(window=int(win), min_periods=int(win)).mean()
        out[sc] = shifted.rolling(window=int(win), min_periods=int(win)).std()
        created.extend([mc, sc])

    for left_lag, right_lag in momentum_pairs:
        col = f"{target_col}_momentum_lag_{left_lag}_{right_lag}"
        out[col] = s.shift(int(left_lag)) - s.shift(int(right_lag))
        created.append(col)

    return out, created


def build_trailing_exogenous_features(
    target_dates: pd.Series,
    exog_df: pd.DataFrame,
    *,
    target_date_col: str,
    exog_date_col: str,
    exog_feature_columns: Sequence[str],
    windows: Sequence[int],
    aggregations: Sequence[str],
    lag_days: int = 0,
    include_missing_indicators: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    aggs = tuple(aggregations)
    invalid = [a for a in aggs if a not in AGG_FUNCS]
    if invalid:
        raise ValueError(f"Unsupported aggregation functions: {invalid}")

    source = exog_df.copy()
    source[exog_date_col] = pd.to_datetime(source[exog_date_col], errors="coerce").dt.normalize()
    source = source.dropna(subset=[exog_date_col]).sort_values(exog_date_col, kind="mergesort")

    rows: list[dict[str, object]] = []
    cols: list[str] = []

    for t in pd.to_datetime(target_dates).normalize():
        row: dict[str, object] = {target_date_col: t}
        cutoff = t - pd.Timedelta(days=lag_days)
        for w in windows:
            start = cutoff - pd.Timedelta(days=int(w) - 1)
            win = source[(source[exog_date_col] >= start) & (source[exog_date_col] <= cutoff)]
            for fcol in exog_feature_columns:
                if fcol not in source.columns:
                    continue
                series = win[fcol]
                for agg in aggs:
                    c = f"{fcol}_{agg}_{w}d"
                    if agg == "mean":
                        row[c] = float(series.mean())
                    elif agg == "sum":
                        row[c] = float(series.fillna(0).sum())
                    elif agg == "max":
                        row[c] = float(series.max())
                    elif agg == "last":
                        row[c] = float(series.iloc[-1]) if len(series) else np.nan
                    elif agg == "count":
                        row[c] = int(series.notna().sum())
                    if c not in cols:
                        cols.append(c)

            event_count_col = f"event_count_{w}d"
            row[event_count_col] = int(len(win))
            if event_count_col not in cols:
                cols.append(event_count_col)

            if include_missing_indicators:
                has_col = f"has_text_signal_{w}d"
                row[has_col] = 1 if len(win) else 0
                if has_col not in cols:
                    cols.append(has_col)

        if include_missing_indicators:
            hist = source[source[exog_date_col] <= cutoff]
            if len(hist):
                row["days_since_last_fed_event"] = int((cutoff - hist[exog_date_col].max()).days)
            else:
                row["days_since_last_fed_event"] = np.nan
            if "days_since_last_fed_event" not in cols:
                cols.append("days_since_last_fed_event")

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(target_date_col, kind="mergesort").reset_index(drop=True)
    return out, cols
