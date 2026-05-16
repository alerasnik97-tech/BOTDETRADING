from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NAME = "mr02_vwap_stretch_reversion"
WARMUP_BARS = 60
EXPLICIT_TIMEFRAME = "M1_OR_M5"


DEFAULT_PARAMS: dict[str, Any] = {
    "session_start": "07:00",
    "session_end": "19:00",
    "band_sd": 2.25,
    "target_band_sd": 0.0,
    "atr_period": 14,
    "stop_atr_buffer": 0.30,
    "min_anchor_bars": 25,
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


def _atr_at(frame: pd.DataFrame, i: int, period: int) -> float:
    if "atr14" in frame.columns:
        value = float(frame["atr14"].iat[i])
        return value if np.isfinite(value) and value > 0 else float("nan")
    if i < period:
        return float("nan")
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
    value = float(true_range.rolling(period, min_periods=period).mean().iat[i])
    return value if np.isfinite(value) and value > 0 else float("nan")


def _previous_session(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> pd.DataFrame:
    ts = frame.index[i]
    previous = frame.iloc[:i]
    mask = [
        idx.date() == ts.date()
        and _in_window(idx, str(params["session_start"]), str(params["session_end"]))
        for idx in previous.index
    ]
    return previous.loc[mask]


def _vwap_stats(window: pd.DataFrame) -> tuple[float, float] | None:
    if window.empty:
        return None
    volume = window["volume"].astype(float)
    close = window["close"].astype(float)
    if (volume <= 0).any():
        return None
    cumulative_volume = volume.cumsum()
    if float(cumulative_volume.iat[-1]) <= 0:
        return None
    vwap_series = (close * volume).cumsum() / cumulative_volume
    vwap = float(vwap_series.iat[-1])
    residual_std = float((close - vwap_series).std(ddof=0))
    if not _all_finite([vwap, residual_std]) or residual_std <= 0:
        return None
    return vwap, residual_std


def _build_signal(
    *,
    direction: str,
    close: float,
    stop_price: float,
    target_price: float,
    session_name: str,
) -> dict[str, Any] | None:
    if direction == "long":
        signal_value = 1
        if stop_price >= close or target_price <= close:
            return None
    else:
        signal_value = -1
        if stop_price <= close or target_price >= close:
            return None
    return {
        "signal": signal_value,
        "direction": direction,
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "price",
        "target_price": target_price,
        "break_even_at_r": None,
        "trailing_atr": False,
        "session_name": session_name,
    }


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    p = {**DEFAULT_PARAMS, **params}
    required = ("open", "high", "low", "close", "volume")
    if i <= 1 or not all(column in frame.columns for column in required):
        return None
    ts = frame.index[i]
    if not _in_window(ts, str(p["session_start"]), str(p["session_end"])):
        return None

    atr = _atr_at(frame, i, int(p["atr_period"]))
    if not np.isfinite(atr) or atr <= 0:
        return None
    window = _previous_session(frame, i, p)
    if len(window) < int(p["min_anchor_bars"]):
        return None
    stats = _vwap_stats(window)
    if stats is None:
        return None
    vwap, sd = stats
    lower = vwap - float(p["band_sd"]) * sd
    upper = vwap + float(p["band_sd"]) * sd
    target_lower = vwap - float(p["target_band_sd"]) * sd
    target_upper = vwap + float(p["target_band_sd"]) * sd

    prev_close = float(frame["close"].iat[i - 1])
    close = float(frame["close"].iat[i])
    low = float(frame["low"].iat[i])
    high = float(frame["high"].iat[i])
    prev_low = float(frame["low"].iat[i - 1])
    prev_high = float(frame["high"].iat[i - 1])
    if not _all_finite([prev_close, close, low, high, prev_low, prev_high, lower, upper]):
        return None

    buffer = float(p["stop_atr_buffer"]) * atr
    if prev_close < lower and close >= lower and close > prev_close:
        return _build_signal(
            direction="long",
            close=close,
            stop_price=min(low, prev_low) - buffer,
            target_price=target_lower,
            session_name=str(p["session_name"]),
        )
    if prev_close > upper and close <= upper and close < prev_close:
        return _build_signal(
            direction="short",
            close=close,
            stop_price=max(high, prev_high) + buffer,
            target_price=target_upper,
            session_name=str(p["session_name"]),
        )
    return None
