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
        "entry_minute_floor": 11 * 60 + 15,
        "latest_signal_minute": 15 * 60 + 45,
        "extremum_lookback": 12,
        "vwap_stretch_std": 1.8,
        "range_atr_min": 1.0,
        "range_atr_max": 2.8,
        "close_reclaim_min": 0.55,
        "stop_buffer_atr": 0.18,
        "target_rr": 1.2,
        "max_hold_bars": 5,
        "h1_adx_max": 24.0,
        "day_range_h1_atr_max": 5.0,
        "h1_ema_distance_max": 8.0,
        "rsi2_long_max": 28.0,
        "rsi2_short_min": 72.0,
        "cooldown_bars": 5,
        "allow_reclaim_after_sweep_bars": 1,
        "break_even_at_r": 0.8,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 15,
        "latest_signal_minute": 15 * 60 + 45,
        "extremum_lookback": 12,
        "vwap_stretch_std": 1.8,
        "range_atr_min": 0.95,
        "range_atr_max": 2.8,
        "close_reclaim_min": 0.55,
        "stop_buffer_atr": 0.18,
        "target_rr": 1.4,
        "max_hold_bars": 5,
        "h1_adx_max": 24.0,
        "day_range_h1_atr_max": 5.0,
        "h1_ema_distance_max": 8.0,
        "rsi2_long_max": 28.0,
        "rsi2_short_min": 72.0,
        "cooldown_bars": 5,
        "allow_reclaim_after_sweep_bars": 2,
        "break_even_at_r": 0.9,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 30,
        "latest_signal_minute": 15 * 60 + 30,
        "extremum_lookback": 12,
        "vwap_stretch_std": 1.85,
        "range_atr_min": 1.0,
        "range_atr_max": 2.6,
        "close_reclaim_min": 0.58,
        "stop_buffer_atr": 0.18,
        "target_rr": 1.2,
        "max_hold_bars": 4,
        "h1_adx_max": 22.0,
        "day_range_h1_atr_max": 4.8,
        "h1_ema_distance_max": 7.0,
        "rsi2_long_max": 27.0,
        "rsi2_short_min": 73.0,
        "cooldown_bars": 5,
        "allow_reclaim_after_sweep_bars": 1,
        "break_even_at_r": 0.8,
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


def _recent_sweep(frame: pd.DataFrame, i: int, lookback: int, side: str, max_age_bars: int) -> bool:
    start = max(lookback, i - int(max_age_bars))
    for j in range(start, i + 1):
        if side == "low":
            if float(frame["low"].iat[j]) <= float(frame["low"].iloc[j - lookback:j].min()):
                return True
        else:
            if float(frame["high"].iat[j]) >= float(frame["high"].iloc[j - lookback:j].max()):
                return True
    return False


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

    reclaim_after_sweep_bars = int(params.get("allow_reclaim_after_sweep_bars", 0))
    close_location = _close_location(frame, i)
    vwap_dist_std = float(frame["vwap_dist_std"].iat[i])
    low_sweep = _recent_sweep(frame, i, lookback, side="low", max_age_bars=reclaim_after_sweep_bars)
    high_sweep = _recent_sweep(frame, i, lookback, side="high", max_age_bars=reclaim_after_sweep_bars)
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
            "break_even_at_r": params.get("break_even_at_r"),
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
            "break_even_at_r": params.get("break_even_at_r"),
            "session_name": str(params["session_name"]),
        }

    return None
