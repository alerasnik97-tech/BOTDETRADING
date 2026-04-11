from __future__ import annotations

import math

import pandas as pd

from research_lab.strategies.common import stratified_sample_combinations


NAME = "rsi_extreme_mean_reversion_core"
WARMUP_BARS = 80
PIP = 0.0001


def parameter_space() -> dict[str, list]:
    return {
        "rsi_period": [14],
        "long_cross_level": [30],
        "short_cross_level": [70],
        "adx_max": [20, 25],
        "stop_pips": [8, 10, 12],
        "tp_mode": ["ema50"],
        "break_even_at_r": [None, 1.0],
        "session_name": ["light_fixed"],
    }


def parameter_grid(max_combinations: int = 12, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def _bar_open_ts(frame: pd.DataFrame, i: int) -> pd.Timestamp:
    return frame.index[i] - pd.Timedelta(minutes=15)


def _setup(frame, i: int, params: dict, direction: str) -> bool:
    rsi_now = float(frame[f"rsi{params['rsi_period']}"].iat[i])
    rsi_prev = float(frame[f"rsi{params['rsi_period']}"].iat[i - 1])
    close = float(frame["close"].iat[i])
    ema50 = float(frame["ema50"].iat[i])
    adx_value = float(frame["adx14"].iat[i])
    if not all(math.isfinite(value) for value in (rsi_now, rsi_prev, close, ema50, adx_value)):
        return False
    if adx_value >= float(params["adx_max"]):
        return False

    if direction == "long":
        return rsi_prev <= float(params["long_cross_level"]) < rsi_now and close < ema50
    return rsi_prev >= float(params["short_cross_level"]) > rsi_now and close > ema50


def _prior_same_direction_today(frame, i: int, params: dict, direction: str) -> bool:
    current_date = _bar_open_ts(frame, i).date()
    j = i - 1
    while j >= 0:
        if _bar_open_ts(frame, j).date() != current_date:
            break
        if _setup(frame, j, params, direction):
            return True
        j -= 1
    return False


def signal(frame, i: int, params: dict) -> dict | None:
    ema50 = float(frame["ema50"].iat[i])
    close = float(frame["close"].iat[i])
    if not all(math.isfinite(value) for value in (ema50, close)):
        return None

    if _setup(frame, i, params, "long") and not _prior_same_direction_today(frame, i, params, "long"):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": close - float(params["stop_pips"]) * PIP,
            "target_mode": "price",
            "target_price": ema50,
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    if _setup(frame, i, params, "short") and not _prior_same_direction_today(frame, i, params, "short"):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": close + float(params["stop_pips"]) * PIP,
            "target_mode": "price",
            "target_price": ema50,
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    return None
