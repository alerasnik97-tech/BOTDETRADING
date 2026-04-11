from __future__ import annotations

import math

import pandas as pd

from research_lab.strategies.common import stratified_sample_combinations


NAME = "ny_fix_momentum_core"
WARMUP_BARS = 80


def parameter_space() -> dict[str, list]:
    return {
        "ema_slope_lookback": [3, 4],
        "min_body_atr_mult": [0.4, 0.5, 0.6],
        "tp_body_mult": [1.2, 1.5, 2.0],
        "break_even_at_r": [None, 1.0],
        "session_name": ["light_fixed"],
    }


def parameter_grid(max_combinations: int = 12, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def _bar_open_ts(frame: pd.DataFrame, i: int) -> pd.Timestamp:
    return frame.index[i] - pd.Timedelta(minutes=15)


def signal(frame, i: int, params: dict) -> dict | None:
    bar_open = _bar_open_ts(frame, i)
    if not (bar_open.hour == 15 and bar_open.minute == 0):
        return None

    atr = float(frame["atr14"].iat[i])
    open_price = float(frame["open"].iat[i])
    close_price = float(frame["close"].iat[i])
    body = abs(close_price - open_price)
    ema50_now = float(frame["ema50"].iat[i])
    ema50_prev = float(frame["ema50"].iat[i - int(params["ema_slope_lookback"])])
    low_price = float(frame["low"].iat[i])
    high_price = float(frame["high"].iat[i])
    if not all(math.isfinite(value) for value in (atr, open_price, close_price, body, ema50_now, ema50_prev, low_price, high_price)):
        return None
    if not (atr > 0 and body > atr * float(params["min_body_atr_mult"])):
        return None

    if close_price > open_price and ema50_now > ema50_prev:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": low_price,
            "target_mode": "price",
            "target_price": close_price + body * float(params["tp_body_mult"]),
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    if close_price < open_price and ema50_now < ema50_prev:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": high_price,
            "target_mode": "price",
            "target_price": close_price - body * float(params["tp_body_mult"]),
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    return None
