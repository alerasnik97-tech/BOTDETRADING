from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.ict_primitives import (
    bearish_displacement,
    bullish_displacement,
    find_recent_sweep_event,
    passes_h1_ema_bias,
    passes_prev_day_premium_discount,
)
from research_lab.strategies.common import is_in_session


NAME = "ict_atomic_sweep_fvg_pm"
WARMUP_BARS = 90
EXPLICIT_TIMEFRAME = "M5"


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 10,
        "latest_signal_minute": 15 * 60 + 40,
        "min_sweep_penetration_pips": 1.0,
        "sweep_to_setup_max_bars": 1,
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
        "target_rr": 1.6,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
        "break_even_at_r": 1.0,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 10,
        "latest_signal_minute": 15 * 60 + 40,
        "min_sweep_penetration_pips": 1.0,
        "sweep_to_setup_max_bars": 1,
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
        "target_rr": 1.6,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
        "break_even_at_r": 1.0,
    },
    {
        "session_name": "pm_11_16",
        "entry_minute_floor": 11 * 60 + 10,
        "latest_signal_minute": 15 * 60 + 40,
        "min_sweep_penetration_pips": 1.0,
        "sweep_to_setup_max_bars": 1,
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
        "target_rr": 1.6,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
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


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 1 or not is_in_session(ts, str(params["session_name"])) or not _time_allowed(ts, params):
        return None

    setup_i = i - 1
    atr_value = float(frame["atr14"].iat[i])
    if atr_value <= 0:
        return None

    if (
        bool(frame["bullish_fvg"].iat[setup_i])
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
            setup_i,
            direction="long",
            min_penetration_pips=float(params["min_sweep_penetration_pips"]),
            max_age_bars=int(params["sweep_to_setup_max_bars"]),
        )
        midpoint = float(frame["bullish_fvg_mid"].iat[setup_i])
        if (
            sweep is not None
            and float(frame["low"].iat[i]) <= midpoint + float(params["retest_buffer_pips"]) * 0.0001
            and float(frame["close"].iat[i]) >= midpoint
            and float(frame["close"].iat[i]) > float(frame["open"].iat[i])
            and _context_passes(frame, i, "long", params)
        ):
            stop_anchor = min(
                float(frame["low"].iat[i]),
                float(frame["low"].iat[setup_i]),
                float(sweep.sweep_price),
                float(sweep.level_price),
            )
            stop_price = stop_anchor - atr_value * float(params["stop_buffer_atr"])
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

    if (
        bool(frame["bearish_fvg"].iat[setup_i])
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
            setup_i,
            direction="short",
            min_penetration_pips=float(params["min_sweep_penetration_pips"]),
            max_age_bars=int(params["sweep_to_setup_max_bars"]),
        )
        midpoint = float(frame["bearish_fvg_mid"].iat[setup_i])
        if (
            sweep is not None
            and float(frame["high"].iat[i]) >= midpoint - float(params["retest_buffer_pips"]) * 0.0001
            and float(frame["close"].iat[i]) <= midpoint
            and float(frame["close"].iat[i]) < float(frame["open"].iat[i])
            and _context_passes(frame, i, "short", params)
        ):
            stop_anchor = max(
                float(frame["high"].iat[i]),
                float(frame["high"].iat[setup_i]),
                float(sweep.sweep_price),
                float(sweep.level_price),
            )
            stop_price = stop_anchor + atr_value * float(params["stop_buffer_atr"])
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
