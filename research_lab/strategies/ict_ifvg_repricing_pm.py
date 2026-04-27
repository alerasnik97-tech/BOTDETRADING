from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.ict_primitives import (
    bearish_displacement,
    bullish_displacement,
    find_recent_ifvg_event,
    passes_h1_ema_bias,
    passes_prev_day_premium_discount,
)
from research_lab.strategies.common import is_in_session


NAME = "ict_ifvg_repricing_pm"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M5"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "main",
        "session_name": "pm_11_1630",
        "entry_minute_floor": 11 * 60 + 10,
        "latest_signal_minute": 15 * 60 + 55,
        "min_fvg_pips": 0.8,
        "min_fvg_atr": 0.10,
        "max_fvg_age_bars": 18,
        "max_inversion_bars": 10,
        "max_retest_bars": 8,
        "require_break_close": True,
        "min_body_atr": 0.60,
        "min_body_fraction": 0.55,
        "min_close_location": 0.70,
        "max_close_location": 0.30,
        "min_range_expansion": 1.10,
        "retest_buffer_pips": 0.25,
        "max_retest_overshoot_pips": 0.80,
        "require_prev_day_pd": False,
        "stop_buffer_atr": 0.12,
        "target_rr": 1.80,
        "max_hold_bars": 8,
        "cooldown_bars": 4,
        "break_even_at_r": 1.0,
    },
    {
        "variant_label": "challenger_prev_day_pd",
        "session_name": "pm_11_1630",
        "entry_minute_floor": 11 * 60 + 10,
        "latest_signal_minute": 15 * 60 + 55,
        "min_fvg_pips": 0.8,
        "min_fvg_atr": 0.10,
        "max_fvg_age_bars": 18,
        "max_inversion_bars": 10,
        "max_retest_bars": 8,
        "require_break_close": True,
        "min_body_atr": 0.60,
        "min_body_fraction": 0.55,
        "min_close_location": 0.70,
        "max_close_location": 0.30,
        "min_range_expansion": 1.10,
        "retest_buffer_pips": 0.25,
        "max_retest_overshoot_pips": 0.80,
        "require_prev_day_pd": True,
        "stop_buffer_atr": 0.12,
        "target_rr": 1.80,
        "max_hold_bars": 8,
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


def _long_context_passes(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> bool:
    if not passes_h1_ema_bias(frame, i, "long"):
        return False
    if bool(params.get("require_prev_day_pd", False)) and not passes_prev_day_premium_discount(frame, i, "long"):
        return False
    return True


def _short_context_passes(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> bool:
    if not passes_h1_ema_bias(frame, i, "short"):
        return False
    if bool(params.get("require_prev_day_pd", False)) and not passes_prev_day_premium_discount(frame, i, "short"):
        return False
    return True


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 2 or not is_in_session(ts, str(params["session_name"])) or not _time_allowed(ts, params):
        return None

    atr_value = float(frame["atr14"].iat[i])
    if atr_value <= 0:
        return None

    long_event = find_recent_ifvg_event(
        frame,
        i - 1,
        direction="long",
        min_fvg_pips=float(params["min_fvg_pips"]),
        min_fvg_atr=float(params["min_fvg_atr"]),
        max_fvg_age_bars=int(params["max_fvg_age_bars"]),
        max_inversion_bars=int(params["max_inversion_bars"]),
        max_retest_bars=int(params["max_retest_bars"]),
        require_break_close=bool(params.get("require_break_close", True)),
    )
    if (
        long_event is not None
        and bullish_displacement(
            frame,
            long_event.inversion_bar_index,
            min_body_atr=float(params["min_body_atr"]),
            min_body_fraction=float(params["min_body_fraction"]),
            min_close_location=float(params["min_close_location"]),
            min_range_expansion=float(params["min_range_expansion"]),
        )
        and _long_context_passes(frame, i, params)
    ):
        overshoot_limit = long_event.bottom - float(params["max_retest_overshoot_pips"]) * PIP_SIZE
        retest_limit = long_event.midpoint + float(params["retest_buffer_pips"]) * PIP_SIZE
        if (
            float(frame["low"].iat[i]) <= retest_limit
            and float(frame["low"].iat[i]) >= overshoot_limit
            and float(frame["close"].iat[i]) >= long_event.midpoint
            and float(frame["close"].iat[i]) > float(frame["open"].iat[i])
        ):
            stop_anchor = min(float(frame["low"].iat[i]), float(long_event.bottom))
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

    short_event = find_recent_ifvg_event(
        frame,
        i - 1,
        direction="short",
        min_fvg_pips=float(params["min_fvg_pips"]),
        min_fvg_atr=float(params["min_fvg_atr"]),
        max_fvg_age_bars=int(params["max_fvg_age_bars"]),
        max_inversion_bars=int(params["max_inversion_bars"]),
        max_retest_bars=int(params["max_retest_bars"]),
        require_break_close=bool(params.get("require_break_close", True)),
    )
    if (
        short_event is not None
        and bearish_displacement(
            frame,
            short_event.inversion_bar_index,
            min_body_atr=float(params["min_body_atr"]),
            min_body_fraction=float(params["min_body_fraction"]),
            max_close_location=float(params["max_close_location"]),
            min_range_expansion=float(params["min_range_expansion"]),
        )
        and _short_context_passes(frame, i, params)
    ):
        overshoot_limit = short_event.top + float(params["max_retest_overshoot_pips"]) * PIP_SIZE
        retest_limit = short_event.midpoint - float(params["retest_buffer_pips"]) * PIP_SIZE
        if (
            float(frame["high"].iat[i]) >= retest_limit
            and float(frame["high"].iat[i]) <= overshoot_limit
            and float(frame["close"].iat[i]) <= short_event.midpoint
            and float(frame["close"].iat[i]) < float(frame["open"].iat[i])
        ):
            stop_anchor = max(float(frame["high"].iat[i]), float(short_event.top))
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
