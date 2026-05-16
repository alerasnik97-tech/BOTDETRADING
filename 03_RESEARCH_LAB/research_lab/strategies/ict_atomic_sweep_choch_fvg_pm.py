from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research_lab.ict_primitives import (
    bearish_displacement,
    bullish_displacement,
    find_recent_sweep_event,
    passes_h1_ema_bias,
    passes_prev_day_premium_discount,
)
from research_lab.strategies.common import is_in_session


NAME = "ict_atomic_sweep_choch_fvg_pm"
WARMUP_BARS = 100
EXPLICIT_TIMEFRAME = "M5"


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 15,
        "latest_signal_minute": 15 * 60 + 35,
        "min_sweep_penetration_pips": 1.0,
        "sweep_to_choch_max_bars": 3,
        "choch_lookback_bars": 4,
        "fvg_search_bars": 2,
        "min_body_atr": 0.75,
        "min_body_fraction": 0.58,
        "min_close_location": 0.68,
        "max_close_location": 0.32,
        "min_range_expansion": 1.25,
        "min_fvg_pips": 0.8,
        "min_fvg_atr": 0.12,
        "retest_buffer_pips": 0.2,
        "require_h1_bias": True,
        "require_prev_day_pd": True,
        "stop_buffer_atr": 0.15,
        "target_rr": 1.8,
        "max_hold_bars": 14,
        "cooldown_bars": 5,
        "break_even_at_r": 1.0,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 15,
        "latest_signal_minute": 15 * 60 + 35,
        "min_sweep_penetration_pips": 1.0,
        "sweep_to_choch_max_bars": 3,
        "choch_lookback_bars": 4,
        "fvg_search_bars": 2,
        "min_body_atr": 0.75,
        "min_body_fraction": 0.58,
        "min_close_location": 0.68,
        "max_close_location": 0.32,
        "min_range_expansion": 1.25,
        "min_fvg_pips": 0.8,
        "min_fvg_atr": 0.12,
        "retest_buffer_pips": 0.2,
        "require_h1_bias": True,
        "require_prev_day_pd": False,
        "stop_buffer_atr": 0.15,
        "target_rr": 1.8,
        "max_hold_bars": 14,
        "cooldown_bars": 5,
        "break_even_at_r": 1.0,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 15,
        "latest_signal_minute": 15 * 60 + 35,
        "min_sweep_penetration_pips": 1.0,
        "sweep_to_choch_max_bars": 3,
        "choch_lookback_bars": 4,
        "fvg_search_bars": 2,
        "min_body_atr": 0.75,
        "min_body_fraction": 0.58,
        "min_close_location": 0.68,
        "max_close_location": 0.32,
        "min_range_expansion": 1.25,
        "min_fvg_pips": 0.8,
        "min_fvg_atr": 0.12,
        "retest_buffer_pips": 0.2,
        "require_h1_bias": False,
        "require_prev_day_pd": False,
        "stop_buffer_atr": 0.15,
        "target_rr": 1.8,
        "max_hold_bars": 14,
        "cooldown_bars": 5,
        "break_even_at_r": 1.0,
    },
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _time_allowed(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    minute_value = ts.hour * 60 + ts.minute
    return int(params["entry_minute_floor"]) <= minute_value <= int(params["latest_signal_minute"])


def _context_passes(frame: pd.DataFrame, i: int, direction: str, params: dict[str, Any]) -> bool:
    if bool(params.get("require_h1_bias", False)) and not passes_h1_ema_bias(frame, i, direction):
        return False
    if bool(params.get("require_prev_day_pd", False)) and not passes_prev_day_premium_discount(frame, i, direction):
        return False
    return True


def _latest_true_index(frame: pd.DataFrame, column: str, start: int, end: int) -> int | None:
    for idx in range(end, start - 1, -1):
        if bool(frame[column].iat[idx]):
            return idx
    return None


def _finite_levels(*levels: float) -> list[float]:
    return [float(level) for level in levels if np.isfinite(float(level))]


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 3 or not is_in_session(ts, str(params["session_name"])) or not _time_allowed(ts, params):
        return None

    atr_value = float(frame["atr14"].iat[i])
    if atr_value <= 0:
        return None

    choch_start = max(0, i - int(params["choch_lookback_bars"]))

    choch_i = _latest_true_index(frame, "bullish_choch", choch_start, i - 1)
    if choch_i is not None:
        setup_i = _latest_true_index(frame, "bullish_fvg", choch_i, min(i - 1, choch_i + int(params["fvg_search_bars"])))
        if (
            setup_i is not None
            and float(frame["bullish_fvg_size_pips"].iat[setup_i]) >= float(params["min_fvg_pips"])
            and float(frame["bullish_fvg_size_atr"].iat[setup_i]) >= float(params["min_fvg_atr"])
            and bullish_displacement(
                frame,
                setup_i,
                min_body_atr=float(params["min_body_atr"]),
                min_body_fraction=float(params["min_body_fraction"]),
                min_close_location=float(params["min_close_location"]),
                min_range_expansion=float(params["min_range_expansion"]),
            )
        ):
            sweep = find_recent_sweep_event(
                frame,
                choch_i,
                direction="long",
                min_penetration_pips=float(params["min_sweep_penetration_pips"]),
                max_age_bars=int(params["sweep_to_choch_max_bars"]),
            )
            midpoint = float(frame["bullish_fvg_mid"].iat[setup_i])
            if (
                sweep is not None
                and float(frame["low"].iat[i]) <= midpoint + float(params["retest_buffer_pips"]) * 0.0001
                and float(frame["close"].iat[i]) >= midpoint
                and float(frame["close"].iat[i]) > float(frame["open"].iat[i])
                and _context_passes(frame, i, "long", params)
            ):
                swing_low = float(frame["last_confirmed_swing_low"].iat[choch_i])
                anchors = _finite_levels(
                    float(frame["low"].iat[i]),
                    float(frame["low"].iat[setup_i]),
                    float(sweep.sweep_price),
                    float(sweep.level_price),
                    swing_low,
                )
                stop_price = min(anchors) - atr_value * float(params["stop_buffer_atr"])
                if stop_price < float(frame["close"].iat[i]):
                    return {
                        "direction": "long",
                        "stop_mode": "price",
                        "stop_price": stop_price,
                        "target_mode": "rr",
                        "target_rr": float(params["target_rr"]),
                        "max_hold_bars": int(params["max_hold_bars"]),
                        "cooldown_bars": int(params.get("cooldown_bars", 0)),
                        "break_even_at_r": params.get("break_even_at_r"),
                        "session_name": str(params["session_name"]),
                    }

    choch_i = _latest_true_index(frame, "bearish_choch", choch_start, i - 1)
    if choch_i is not None:
        setup_i = _latest_true_index(frame, "bearish_fvg", choch_i, min(i - 1, choch_i + int(params["fvg_search_bars"])))
        if (
            setup_i is not None
            and float(frame["bearish_fvg_size_pips"].iat[setup_i]) >= float(params["min_fvg_pips"])
            and float(frame["bearish_fvg_size_atr"].iat[setup_i]) >= float(params["min_fvg_atr"])
            and bearish_displacement(
                frame,
                setup_i,
                min_body_atr=float(params["min_body_atr"]),
                min_body_fraction=float(params["min_body_fraction"]),
                max_close_location=float(params["max_close_location"]),
                min_range_expansion=float(params["min_range_expansion"]),
            )
        ):
            sweep = find_recent_sweep_event(
                frame,
                choch_i,
                direction="short",
                min_penetration_pips=float(params["min_sweep_penetration_pips"]),
                max_age_bars=int(params["sweep_to_choch_max_bars"]),
            )
            midpoint = float(frame["bearish_fvg_mid"].iat[setup_i])
            if (
                sweep is not None
                and float(frame["high"].iat[i]) >= midpoint - float(params["retest_buffer_pips"]) * 0.0001
                and float(frame["close"].iat[i]) <= midpoint
                and float(frame["close"].iat[i]) < float(frame["open"].iat[i])
                and _context_passes(frame, i, "short", params)
            ):
                swing_high = float(frame["last_confirmed_swing_high"].iat[choch_i])
                anchors = _finite_levels(
                    float(frame["high"].iat[i]),
                    float(frame["high"].iat[setup_i]),
                    float(sweep.sweep_price),
                    float(sweep.level_price),
                    swing_high,
                )
                stop_price = max(anchors) + atr_value * float(params["stop_buffer_atr"])
                if stop_price > float(frame["close"].iat[i]):
                    return {
                        "direction": "short",
                        "stop_mode": "price",
                        "stop_price": stop_price,
                        "target_mode": "rr",
                        "target_rr": float(params["target_rr"]),
                        "max_hold_bars": int(params["max_hold_bars"]),
                        "cooldown_bars": int(params.get("cooldown_bars", 0)),
                        "break_even_at_r": params.get("break_even_at_r"),
                        "session_name": str(params["session_name"]),
                    }

    return None
