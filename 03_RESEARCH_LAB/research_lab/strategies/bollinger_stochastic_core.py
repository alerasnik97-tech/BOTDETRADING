from __future__ import annotations

import math

import pandas as pd

from research_lab.strategies.common import stratified_sample_combinations


NAME = "bollinger_stochastic_core"
WARMUP_BARS = 80
PIP = 0.0001


def parameter_space() -> dict[str, list]:
    return {
        "bb_period": [20],
        "bb_std": [2.0, 2.2],
        "stoch_k": [14],
        "stoch_smooth": [3],
        "adx_max": [18, 20],
        "stop_buffer_pips": [2],
        "target_mode": ["midband"],
        "break_even_at_r": [None, 1.0],
        "session_name": ["light_fixed"],
    }


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def _bar_open_ts(frame: pd.DataFrame, i: int) -> pd.Timestamp:
    return frame.index[i] - pd.Timedelta(minutes=15)


def _time_allowed(frame: pd.DataFrame, i: int) -> bool:
    ts = _bar_open_ts(frame, i)
    minute = ts.hour * 60 + ts.minute
    in_lunch = 11 * 60 <= minute < 13 * 60
    in_late = 17 * 60 <= minute < 19 * 60
    return in_lunch or in_late


def _setup(frame: pd.DataFrame, i: int, params: dict, direction: str) -> bool:
    if not _time_allowed(frame, i):
        return False
    adx_value = float(frame["adx14"].iat[i])
    if not math.isfinite(adx_value) or adx_value >= float(params["adx_max"]):
        return False

    suffix = f"{params['bb_period']}_{str(params['bb_std']).replace('.', '_')}"
    lower = float(frame[f"bb_lower_{suffix}"].iat[i])
    upper = float(frame[f"bb_upper_{suffix}"].iat[i])
    stoch_k = float(frame[f"stoch_k_{params['stoch_k']}_{params['stoch_smooth']}_3"].iat[i])
    close = float(frame["close"].iat[i])
    if not all(math.isfinite(value) for value in (lower, upper, stoch_k, close)):
        return False

    if direction == "long":
        return close < lower and stoch_k < 20.0
    return close > upper and stoch_k > 80.0


def _prior_same_direction_today(frame: pd.DataFrame, i: int, params: dict, direction: str) -> bool:
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
    suffix = f"{params['bb_period']}_{str(params['bb_std']).replace('.', '_')}"
    mid = float(frame[f"bb_mid_{suffix}"].iat[i])
    lower = float(frame[f"bb_lower_{suffix}"].iat[i])
    upper = float(frame[f"bb_upper_{suffix}"].iat[i])
    if not all(math.isfinite(value) for value in (mid, lower, upper)):
        return None

    if _setup(frame, i, params, "long") and not _prior_same_direction_today(frame, i, params, "long"):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": lower - float(params["stop_buffer_pips"]) * PIP,
            "target_mode": "price",
            "target_price": mid,
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    if _setup(frame, i, params, "short") and not _prior_same_direction_today(frame, i, params, "short"):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": upper + float(params["stop_buffer_pips"]) * PIP,
            "target_mode": "price",
            "target_price": mid,
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }

    return None
