from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NAME = "ve_orb_volatility_expansion"
WARMUP_BARS = 220
EXPLICIT_TIMEFRAME = "M1_OR_M5"


DEFAULT_PARAMS: dict[str, Any] = {
    "or_start": "07:00",
    "or_end": "08:00",
    "entry_end": "12:00",
    "atr_period": 14,
    "atr_percentile_lookback": 200,
    "atr_percentile": 65.0,
    "min_or_atr": 0.40,
    "max_or_atr": 3.00,
    "min_or_coverage_pct": 0.90,
    "allow_inferred_timeframe": True,
    "min_or_bars": None,
    "target_rr": 2.0,
    "session_name": "all_day",
}


def parameter_space() -> dict[str, list[Any]]:
    return {key: [value] for key, value in DEFAULT_PARAMS.items()}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return [default_params() for _ in range(max(1, max_combinations))][:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_PARAMS)


def _minute(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _minute_of(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _all_finite(values: list[float]) -> bool:
    return all(np.isfinite(value) for value in values)


def _atr_series(frame: pd.DataFrame, period: int) -> pd.Series:
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    closes = frame["close"].astype(float)
    prev_closes = closes.shift(1)
    true_range = pd.concat(
        [
            highs - lows,
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _infer_cadence_minutes(index: pd.DatetimeIndex) -> int | None:
    if len(index) < 3 or index.has_duplicates:
        return None
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    positive = deltas[deltas > 0]
    if positive.empty:
        return None
    cadence = float(positive.median())
    rounded = int(round(cadence))
    if abs(cadence - rounded) > 0.01 or rounded not in {1, 2, 3, 5, 10, 15}:
        return None
    return rounded


def _or_window_is_complete(window: pd.DataFrame, rows: pd.DataFrame, params: dict[str, Any]) -> bool:
    if window.empty or window.index.has_duplicates:
        return False
    if not bool(params.get("allow_inferred_timeframe", True)):
        return False

    cadence = _infer_cadence_minutes(pd.DatetimeIndex(rows.index))
    if cadence is None:
        return False

    expected_bars = int(np.floor((_minute(str(params["or_end"])) - _minute(str(params["or_start"]))) / cadence))
    explicit_min = params.get("min_or_bars")
    if explicit_min is not None:
        expected_bars = max(expected_bars, int(explicit_min))
    if expected_bars < 2:
        return False

    required_bars = int(np.ceil(expected_bars * float(params["min_or_coverage_pct"])))
    unique_count = len(pd.DatetimeIndex(window.index).unique())
    if unique_count < max(2, required_bars):
        return False

    first_minute = _minute_of(window.index.min())
    last_minute = _minute_of(window.index.max())
    start = _minute(str(params["or_start"]))
    end = _minute(str(params["or_end"]))
    if first_minute > start + cadence:
        return False
    if last_minute < end - cadence:
        return False
    return True


def _opening_range(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> tuple[float, float] | None:
    ts = frame.index[i]
    start = _minute(str(params["or_start"]))
    end = _minute(str(params["or_end"]))
    rows = frame.iloc[:i]
    mask = [
        idx.date() == ts.date() and start <= _minute_of(idx) < end
        for idx in rows.index
    ]
    window = rows.loc[mask]
    if not _or_window_is_complete(window, rows, params):
        return None
    if window[["high", "low", "close"]].isna().any().any():
        return None
    high = float(window["high"].max())
    low = float(window["low"].min())
    if not _all_finite([high, low]) or high <= low:
        return None
    return high, low


def _build_signal(
    *,
    direction: str,
    close: float,
    stop_price: float,
    target_rr: float,
    session_name: str,
) -> dict[str, Any] | None:
    if direction == "long":
        signal_value = 1
        if stop_price >= close:
            return None
    else:
        signal_value = -1
        if stop_price <= close:
            return None
    return {
        "signal": signal_value,
        "direction": direction,
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "rr",
        "target_rr": target_rr,
        "break_even_at_r": None,
        "trailing_atr": False,
        "session_name": session_name,
    }


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    p = {**DEFAULT_PARAMS, **params}
    required = ("open", "high", "low", "close")
    if i <= 1 or not all(column in frame.columns for column in required):
        return None

    current_minute = _minute_of(frame.index[i])
    if current_minute < _minute(str(p["or_end"])) or current_minute >= _minute(str(p["entry_end"])):
        return None

    atr_period = int(p["atr_period"])
    lookback = int(p["atr_percentile_lookback"])
    if i < lookback + atr_period:
        return None

    atr_values = _atr_series(frame, atr_period)
    current_atr = float(atr_values.iat[i])
    previous_atr_window = atr_values.iloc[i - lookback : i].dropna()
    if len(previous_atr_window) < lookback:
        return None
    threshold = float(np.percentile(previous_atr_window.to_numpy(dtype=float), float(p["atr_percentile"])))
    if not _all_finite([current_atr, threshold]) or current_atr <= threshold or current_atr <= 0:
        return None

    opening_range = _opening_range(frame, i, p)
    if opening_range is None:
        return None
    or_high, or_low = opening_range
    or_atr = (or_high - or_low) / current_atr
    if or_atr < float(p["min_or_atr"]) or or_atr > float(p["max_or_atr"]):
        return None

    prev_close = float(frame["close"].iat[i - 1])
    close = float(frame["close"].iat[i])
    if not _all_finite([prev_close, close, or_high, or_low]):
        return None

    if prev_close <= or_high < close:
        return _build_signal(
            direction="long",
            close=close,
            stop_price=or_low,
            target_rr=float(p["target_rr"]),
            session_name=str(p["session_name"]),
        )
    if prev_close >= or_low > close:
        return _build_signal(
            direction="short",
            close=close,
            stop_price=or_high,
            target_rr=float(p["target_rr"]),
            session_name=str(p["session_name"]),
        )
    return None
