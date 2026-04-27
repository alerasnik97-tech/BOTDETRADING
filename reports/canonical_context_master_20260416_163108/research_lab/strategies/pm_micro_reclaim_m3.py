from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import is_in_session


NAME = "pm_micro_reclaim_m3"
WARMUP_BARS = 40
EXPLICIT_TIMEFRAME = "M3"


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 45,
        "extremum_lookback": 20,
        "vwap_stretch_std": 1.8,
        "range_atr_min": 1.0,
        "range_atr_max": 2.6,
        "close_reclaim_min": 0.60,
        "stop_buffer_atr": 0.15,
        "target_rr": 1.2,
        "max_hold_bars": 5,
        "h1_adx_max": 22.0,
        "day_range_h1_atr_max": 5.0,
        "h1_ema_distance_max": 8.0,
        "rsi2_long_max": 25.0,
        "rsi2_short_min": 75.0,
        "cooldown_bars": 6,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 45,
        "extremum_lookback": 20,
        "vwap_stretch_std": 1.9,
        "range_atr_min": 1.0,
        "range_atr_max": 2.8,
        "close_reclaim_min": 0.60,
        "stop_buffer_atr": 0.15,
        "target_rr": 1.4,
        "max_hold_bars": 6,
        "h1_adx_max": 22.0,
        "day_range_h1_atr_max": 5.0,
        "h1_ema_distance_max": 8.0,
        "rsi2_long_max": 22.0,
        "rsi2_short_min": 78.0,
        "cooldown_bars": 6,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 45,
        "extremum_lookback": 12,
        "vwap_stretch_std": 1.7,
        "range_atr_min": 0.9,
        "range_atr_max": 2.4,
        "close_reclaim_min": 0.60,
        "stop_buffer_atr": 0.20,
        "target_rr": 1.2,
        "max_hold_bars": 5,
        "h1_adx_max": 20.0,
        "day_range_h1_atr_max": 4.5,
        "h1_ema_distance_max": 6.0,
        "rsi2_long_max": 28.0,
        "rsi2_short_min": 72.0,
        "cooldown_bars": 6,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 45,
        "extremum_lookback": 12,
        "vwap_stretch_std": 1.9,
        "range_atr_min": 1.1,
        "range_atr_max": 2.8,
        "close_reclaim_min": 0.60,
        "stop_buffer_atr": 0.20,
        "target_rr": 1.4,
        "max_hold_bars": 6,
        "h1_adx_max": 20.0,
        "day_range_h1_atr_max": 4.5,
        "h1_ema_distance_max": 6.0,
        "rsi2_long_max": 25.0,
        "rsi2_short_min": 75.0,
        "cooldown_bars": 6,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 30,
        "extremum_lookback": 20,
        "vwap_stretch_std": 1.6,
        "range_atr_min": 0.8,
        "range_atr_max": 2.2,
        "close_reclaim_min": 0.60,
        "stop_buffer_atr": 0.12,
        "target_rr": 1.0,
        "max_hold_bars": 4,
        "h1_adx_max": 18.0,
        "day_range_h1_atr_max": 4.0,
        "h1_ema_distance_max": 5.0,
        "rsi2_long_max": 28.0,
        "rsi2_short_min": 80.0,
        "cooldown_bars": 5,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 30,
        "extremum_lookback": 20,
        "vwap_stretch_std": 1.8,
        "range_atr_min": 0.8,
        "range_atr_max": 2.4,
        "close_reclaim_min": 0.60,
        "stop_buffer_atr": 0.12,
        "target_rr": 1.2,
        "max_hold_bars": 4,
        "h1_adx_max": 18.0,
        "day_range_h1_atr_max": 4.0,
        "h1_ema_distance_max": 5.0,
        "rsi2_long_max": 25.0,
        "rsi2_short_min": 78.0,
        "cooldown_bars": 5,
    },
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _close_location(frame: pd.DataFrame, i: int) -> float:
    bar_range = float(frame["high"].iat[i] - frame["low"].iat[i])
    if bar_range <= 0:
        return 0.5
    return float((frame["close"].iat[i] - frame["low"].iat[i]) / bar_range)


def _time_allowed(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    minute_value = ts.hour * 60 + ts.minute
    return int(params["entry_minute_floor"]) <= minute_value <= int(params["latest_signal_minute"])


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if not is_in_session(ts, str(params["session_name"])):
        return None
    if not _time_allowed(ts, params):
        return None

    lookback = int(params["extremum_lookback"])
    if i < lookback:
        return None

    atr_value = float(frame["atr14"].iat[i])
    if atr_value <= 0:
        return None
    range_atr = float(frame["range_atr"].iat[i])
    if range_atr < float(params["range_atr_min"]) or range_atr > float(params["range_atr_max"]):
        return None
    if float(frame["h1_adx14"].iat[i]) > float(params["h1_adx_max"]):
        return None
    if float(frame["day_range_h1_atr"].iat[i]) > float(params["day_range_h1_atr_max"]):
        return None
    h1_atr = float(frame["h1_atr14"].iat[i])
    if h1_atr <= 0:
        return None
    h1_ema_distance = abs(float(frame["close"].iat[i] - frame["h1_ema200"].iat[i])) / h1_atr
    if h1_ema_distance > float(params["h1_ema_distance_max"]):
        return None

    close_location = _close_location(frame, i)
    vwap_dist_std = float(frame["vwap_dist_std"].iat[i])
    low_sweep = float(frame["low"].iat[i]) <= float(frame["low"].iloc[i - lookback:i].min())
    high_sweep = float(frame["high"].iat[i]) >= float(frame["high"].iloc[i - lookback:i].max())
    bullish_reclaim = float(frame["close"].iat[i]) > float(frame["open"].iat[i]) and close_location >= float(params["close_reclaim_min"])
    bearish_reclaim = float(frame["close"].iat[i]) < float(frame["open"].iat[i]) and close_location <= 1.0 - float(params["close_reclaim_min"])

    if (
        low_sweep
        and bullish_reclaim
        and vwap_dist_std <= -float(params["vwap_stretch_std"])
        and float(frame["rsi2"].iat[i]) <= float(params["rsi2_long_max"])
    ):
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": float(frame["low"].iat[i] - atr_value * float(params["stop_buffer_atr"])),
            "target_mode": "rr",
            "target_rr": float(params["target_rr"]),
            "max_hold_bars": int(params["max_hold_bars"]),
            "cooldown_bars": int(params.get("cooldown_bars", 0)),
            "session_name": str(params["session_name"]),
        }

    if (
        high_sweep
        and bearish_reclaim
        and vwap_dist_std >= float(params["vwap_stretch_std"])
        and float(frame["rsi2"].iat[i]) >= float(params["rsi2_short_min"])
    ):
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": float(frame["high"].iat[i] + atr_value * float(params["stop_buffer_atr"])),
            "target_mode": "rr",
            "target_rr": float(params["target_rr"]),
            "max_hold_bars": int(params["max_hold_bars"]),
            "cooldown_bars": int(params.get("cooldown_bars", 0)),
            "session_name": str(params["session_name"]),
        }

    return None
