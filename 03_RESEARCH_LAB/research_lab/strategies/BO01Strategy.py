from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ID = "BO01"
FAMILY_ID = "LBC"
NAME = "BO01Strategy"
WARMUP_BARS = 80
EXPLICIT_TIMEFRAME = "M5"

ASIAN_MIN_BARS = 79

DEFAULT_PARAMS: dict[str, Any] = {
    "asian_start": "00:00",
    "asian_end": "06:30",
    "entry_start": "07:00",
    "entry_end": "10:00",
    "atr_period": 14,
    "atr_multiplier": 0.5,
    "ema_trend_period": 20,
    "min_asian_range_pips": 8.0,
    "pip_size": 0.0001,
    "target_rr": 2.0,
    "daily_trade_count": 0,
    "has_active_position": False,
    "session_name": "london",
}


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_PARAMS)


def parameter_space() -> dict[str, list[Any]]:
    return {key: [value] for key, value in DEFAULT_PARAMS.items()}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return [default_params() for _ in range(max(1, max_combinations))][:max_combinations]


def _minute(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _utc_ts(ts: pd.Timestamp) -> pd.Timestamp | None:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return None
    return stamp.tz_convert("UTC")


def _utc_minute(ts: pd.Timestamp) -> int | None:
    stamp = _utc_ts(ts)
    if stamp is None:
        return None
    return stamp.hour * 60 + stamp.minute


def _all_finite(values: list[float]) -> bool:
    return all(np.isfinite(value) for value in values)


def _required_columns_present(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(column in frame.columns for column in columns)


def _atr_at(frame: pd.DataFrame, i: int, period: int) -> float:
    if "atr14" in frame.columns:
        value = float(frame["atr14"].iat[i])
        return value if np.isfinite(value) and value > 0 else float("nan")
    if i < period:
        return float("nan")
    rows = frame.iloc[: i + 1]
    highs = rows["high"].astype(float)
    lows = rows["low"].astype(float)
    closes = rows["close"].astype(float)
    prev_closes = closes.shift(1)
    true_range = pd.concat(
        [
            highs - lows,
            (highs - prev_closes).abs(),
            (lows - prev_closes).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(true_range.rolling(period, min_periods=period).mean().iat[-1])
    return value if np.isfinite(value) and value > 0 else float("nan")


def _ema_at(frame: pd.DataFrame, i: int, period: int) -> float:
    if i + 1 < period:
        return float("nan")
    values = frame["close"].iloc[: i + 1].astype(float)
    if values.isna().any() or not np.isfinite(values).all():
        return float("nan")
    value = float(values.ewm(span=period, adjust=False, min_periods=period).mean().iat[-1])
    return value if np.isfinite(value) else float("nan")


def _asian_range(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> tuple[float, float] | None:
    ts = _utc_ts(frame.index[i])
    if ts is None:
        return None
    start = _minute(str(params["asian_start"]))
    end = _minute(str(params["asian_end"]))
    rows = frame.iloc[:i]
    selected: list[int] = []
    for idx, value in enumerate(rows.index):
        stamp = _utc_ts(value)
        if stamp is None:
            return None
        minute = stamp.hour * 60 + stamp.minute
        if stamp.date() == ts.date() and start <= minute <= end:
            selected.append(idx)
    if len(selected) < ASIAN_MIN_BARS:
        return None
    window = rows.iloc[selected]
    if window[["high", "low"]].isna().any().any():
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
    required = ("open", "high", "low", "close", "volume", "ema_m15_200")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        return None
    if i < WARMUP_BARS or i >= len(frame) or not _required_columns_present(frame, required):
        return None
    if int(p.get("daily_trade_count", 0)) > 0 or bool(p.get("has_active_position", False)):
        return None

    minute = _utc_minute(frame.index[i])
    if minute is None or minute < _minute(str(p["entry_start"])) or minute > _minute(str(p["entry_end"])):
        return None

    current_values = [
        float(frame["open"].iat[i]),
        float(frame["high"].iat[i]),
        float(frame["low"].iat[i]),
        float(frame["close"].iat[i]),
        float(frame["volume"].iat[i]),
        float(frame["ema_m15_200"].iat[i]),
    ]
    if not _all_finite(current_values):
        return None

    asian = _asian_range(frame, i, p)
    if asian is None:
        return None
    asian_high, asian_low = asian
    pip_size = float(p["pip_size"])
    width_pips = (asian_high - asian_low) / pip_size
    if width_pips < float(p["min_asian_range_pips"]):
        return None

    atr = _atr_at(frame, i, int(p["atr_period"]))
    ema_m5 = _ema_at(frame, i, int(p["ema_trend_period"]))
    close = float(frame["close"].iat[i])
    ema_m15 = float(frame["ema_m15_200"].iat[i])
    if not _all_finite([atr, ema_m5, close, ema_m15]) or atr <= 0:
        return None

    threshold = float(p["atr_multiplier"]) * atr
    midpoint = asian_low + (asian_high - asian_low) / 2.0
    if close > asian_high + threshold and close > ema_m5 and close > ema_m15:
        return _build_signal(
            direction="long",
            close=close,
            stop_price=midpoint,
            target_rr=float(p["target_rr"]),
            session_name=str(p["session_name"]),
        )
    if close < asian_low - threshold and close < ema_m5 and close < ema_m15:
        return _build_signal(
            direction="short",
            close=close,
            stop_price=midpoint,
            target_rr=float(p["target_rr"]),
            session_name=str(p["session_name"]),
        )
    return None
