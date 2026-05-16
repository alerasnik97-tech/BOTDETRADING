from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NAME = "tp01_london_ny_momentum_pullback"
WARMUP_BARS = 220
EXPLICIT_TIMEFRAME = "M1_OR_M5"


DEFAULT_PARAMS: dict[str, Any] = {
    "entry_start": "08:00",
    "entry_end": "12:00",
    "ema_period": 20,
    "atr_period": 14,
    "atr_percentile_lookback": 200,
    "atr_percentile": 50.0,
    "momentum_bars": 5,
    "momentum_atr_mult": 1.5,
    "pullback_tolerance_atr": 0.25,
    "stop_atr_buffer": 0.35,
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


def _in_window(ts: pd.Timestamp, start: str, end: str) -> bool:
    minute = _minute_of(ts)
    return _minute(start) <= minute < _minute(end)


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


_CACHE: dict[tuple[int, int, Any, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_cached_indicators(frame: pd.DataFrame, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    p = {**DEFAULT_PARAMS, **params}
    atr_period = int(p["atr_period"])
    ema_period = int(p["ema_period"])
    
    # Cache key based on frame id, frame shape/length, last index timestamp, and parameters
    key = (id(frame), len(frame), frame.index[-1] if len(frame) > 0 else None, atr_period, ema_period)
    
    if key not in _CACHE:
        # Precompute indicators safely as numpy float64 arrays
        atr_series = _atr_series(frame, atr_period)
        atr_values = atr_series.to_numpy(dtype=float)
        close_series = frame["close"].astype(float)
        ema_values = close_series.ewm(span=ema_period, adjust=False).mean().to_numpy(dtype=float)
        _CACHE[key] = (atr_values, ema_values)
        
    return _CACHE[key]


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    p = {**DEFAULT_PARAMS, **params}
    required = ("open", "high", "low", "close")
    if i <= 1 or not all(column in frame.columns for column in required):
        return None
    if not _in_window(frame.index[i], str(p["entry_start"]), str(p["entry_end"])):
        return None

    atr_period = int(p["atr_period"])
    lookback = int(p["atr_percentile_lookback"])
    momentum_bars = int(p["momentum_bars"])
    ema_period = int(p["ema_period"])
    
    if i < lookback + atr_period + momentum_bars:
        return None
    if i < ema_period + 2:
        return None

    # Retrieve precomputed indicator arrays
    atr_values, ema_values = _get_cached_indicators(frame, p)

    current_atr = float(atr_values[i])
    # Directly slice numpy array view - extremely fast, no NaNs possible for i >= lookback + atr_period + momentum_bars
    previous_atr_window = atr_values[i - lookback : i]
    threshold = float(np.percentile(previous_atr_window, float(p["atr_percentile"])))
    if not _all_finite([current_atr, threshold]) or current_atr <= threshold or current_atr <= 0:
        return None

    ema_now = float(ema_values[i - 1])
    ema_prev = float(ema_values[i - 2])

    close = float(frame["close"].iat[i])
    prev_close = float(frame["close"].iat[i - 1])
    close_before_momentum = float(frame["close"].iat[i - momentum_bars - 1])
    momentum = prev_close - close_before_momentum
    required_momentum = float(p["momentum_atr_mult"]) * current_atr
    low = float(frame["low"].iat[i])
    high = float(frame["high"].iat[i])
    prev_high = float(frame["high"].iat[i - 1])
    prev_low = float(frame["low"].iat[i - 1])
    tolerance = float(p["pullback_tolerance_atr"]) * current_atr
    buffer = float(p["stop_atr_buffer"]) * current_atr
    if not _all_finite([ema_now, ema_prev, close, prev_close, momentum, low, high, prev_high, prev_low]):
        return None

    long_bias = momentum > required_momentum and prev_close > ema_now and ema_now >= ema_prev
    short_bias = momentum < -required_momentum and prev_close < ema_now and ema_now <= ema_prev

    if long_bias and low <= ema_now + tolerance and close > ema_now and close > prev_high:
        return _build_signal(
            direction="long",
            close=close,
            stop_price=min(low, prev_low) - buffer,
            target_rr=float(p["target_rr"]),
            session_name=str(p["session_name"]),
        )
    if short_bias and high >= ema_now - tolerance and close < ema_now and close < prev_low:
        return _build_signal(
            direction="short",
            close=close,
            stop_price=max(high, prev_high) + buffer,
            target_rr=float(p["target_rr"]),
            session_name=str(p["session_name"]),
        )
    return None
