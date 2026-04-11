from __future__ import annotations

import math

import pandas as pd

from research_lab.strategies.common import stratified_sample_combinations


NAME = "macd_3_10_16_pullback_core"
WARMUP_BARS = 80


def parameter_space() -> dict[str, list]:
    return {
        "ema_trend_lookback": [4, 5, 6],
        "rsi_period": [7],
        "rsi_min": [35, 40],
        "rsi_max": [60, 65],
        "swing_lookback": [3],
        "target_rr": [1.5, 2.0],
        "break_even_at_r": [None, 1.0],
        "session_name": ["light_fixed"],
    }


def parameter_grid(max_combinations: int = 12, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def _bar_open_ts(frame: pd.DataFrame, i: int) -> pd.Timestamp:
    return frame.index[i] - pd.Timedelta(minutes=15)


def _time_allowed(frame: pd.DataFrame, i: int) -> bool:
    ts = _bar_open_ts(frame, i)
    minute = ts.hour * 60 + ts.minute
    return 11 * 60 <= minute < 18 * 60


def signal(frame, i: int, params: dict) -> dict | None:
    if not _time_allowed(frame, i):
        return None

    ema50_now = float(frame["ema50"].iat[i])
    ema50_prev = float(frame["ema50"].iat[i - int(params["ema_trend_lookback"])])
    hist_now = float(frame["macd_hist_3_10_16"].iat[i])
    hist_prev = float(frame["macd_hist_3_10_16"].iat[i - 1])
    rsi_value = float(frame[f"rsi{params['rsi_period']}"].iat[i])
    close = float(frame["close"].iat[i])
    if not all(math.isfinite(value) for value in (ema50_now, ema50_prev, hist_now, hist_prev, rsi_value, close)):
        return None

    rsi_ok = float(params["rsi_min"]) <= rsi_value <= float(params["rsi_max"])
    if not rsi_ok:
        return None

    lookback = int(params["swing_lookback"])
    swing_low = float(frame["low"].iloc[i - lookback + 1 : i + 1].min())
    swing_high = float(frame["high"].iloc[i - lookback + 1 : i + 1].max())

    long_signal = ema50_now > ema50_prev and hist_prev <= 0.0 < hist_now and close > ema50_now
    short_signal = ema50_now < ema50_prev and hist_prev >= 0.0 > hist_now and close < ema50_now

    if long_signal and math.isfinite(swing_low):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": swing_low,
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    if short_signal and math.isfinite(swing_high):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": swing_high,
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    return None
