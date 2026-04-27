from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.ict_primitives import bearish_displacement, find_recent_sweep_event
from research_lab.strategies.common import finite, is_in_session


NAME = "eurusd_h1_liquidity_sweep_m15"
WARMUP_BARS = 80
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001
LEVEL_COLUMNS = ("prev_week_high", "prev_day_high")


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "upper_htf_raid_reject_v1",
        "session_name": "research_08_1630",
        "min_sweep_pips": 1.0,
        "max_sweep_age_bars": 2,
        "confirmation_window_bars": 2,
        "min_body_atr": 0.25,
        "min_body_fraction": 0.45,
        "max_close_location": 0.35,
        "min_range_expansion": 1.0,
        "stop_buffer_pips": 0.5,
        "target_rr": 1.0,
        "max_hold_bars": 8,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _first_break_after_sweep(frame: pd.DataFrame, *, sweep_bar_index: int, confirmation_index: int, sweep_low: float) -> bool:
    if confirmation_index <= sweep_bar_index:
        return False
    if confirmation_index == sweep_bar_index + 1:
        return True
    window = frame.iloc[sweep_bar_index + 1 : confirmation_index]
    if window.empty:
        return True
    return not bool((window["close"] < sweep_low).any())


def _build_signal(
    *,
    params: dict[str, Any],
    signal_price: float,
    stop_price: float,
) -> dict[str, Any]:
    return {
        "direction": "short",
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "rr",
        "target_rr": float(params["target_rr"]),
        "max_hold_bars": int(params["max_hold_bars"]),
        "session_name": str(params["session_name"]),
        "signal_price": signal_price,
    }


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    if i < 3:
        return None

    ts = frame.index[i]
    if not is_in_session(ts, str(params["session_name"])):
        return None

    sweep = find_recent_sweep_event(
        frame,
        i,
        direction="short",
        min_penetration_pips=float(params["min_sweep_pips"]),
        max_age_bars=int(params["max_sweep_age_bars"]),
        level_columns=LEVEL_COLUMNS,
        pip_size=PIP_SIZE,
    )
    if sweep is None:
        return None

    bars_since_sweep = i - int(sweep.bar_index)
    if bars_since_sweep < 1 or bars_since_sweep > int(params["confirmation_window_bars"]):
        return None

    sweep_low = float(frame["low"].iat[sweep.bar_index])
    if not _first_break_after_sweep(frame, sweep_bar_index=int(sweep.bar_index), confirmation_index=i, sweep_low=sweep_low):
        return None

    if float(frame["close"].iat[i]) >= sweep_low:
        return None

    if not bearish_displacement(
        frame,
        i,
        min_body_atr=float(params["min_body_atr"]),
        min_body_fraction=float(params["min_body_fraction"]),
        max_close_location=float(params["max_close_location"]),
        min_range_expansion=float(params["min_range_expansion"]),
    ):
        return None

    signal_price = float(frame["close"].iat[i])
    stop_price = max(float(frame["high"].iat[sweep.bar_index]), float(frame["high"].iat[i])) + (float(params["stop_buffer_pips"]) * PIP_SIZE)
    if not finite(signal_price) or not finite(stop_price) or stop_price <= signal_price:
        return None

    return _build_signal(params=params, signal_price=signal_price, stop_price=stop_price)
