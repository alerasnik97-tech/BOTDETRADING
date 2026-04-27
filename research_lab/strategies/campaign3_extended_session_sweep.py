from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.ict_primitives import add_pivot_structure_columns
from research_lab.strategies.common import is_in_session, session_window


NAME = "campaign3_extended_session_sweep"
WARMUP_BARS = 150
EXPLICIT_TIMEFRAME = "M5"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_08_11",
        "session_name": "am_08_11",
        "entry_start": "08:00",
        "entry_end": "11:00",
        "force_close": "11:30",
        "target_rr": 2.1,
        "max_hold_bars": 21,
        "cooldown_bars": 0,
        "break_even_at_r": 1.2,
        "sweep_extension_atr": 0.25,
        "displacement_body_atr": 0.25,
        "displacement_body_fraction": 0.45,
        "displacement_close_location_long": 0.65,
        "displacement_close_location_short": 0.35,
        "confirmation_bars": 6,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _is_time_to_trade(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    minute_value = ts.hour * 60 + ts.minute
    start_h, start_m = map(int, str(params["entry_start"]).split(":"))
    end_h, end_m = map(int, str(params["entry_end"]).split(":"))
    return (start_h * 60 + start_m) <= minute_value < (end_h * 60 + end_m)


def _is_past_force_close(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    force_close = str(params.get("force_close", "11:30"))
    force_h, force_m = map(int, force_close.split(":"))
    force_total = force_h * 60 + force_m
    curr_total = ts.hour * 60 + ts.minute
    return curr_total >= force_total


def _get_liquidity_levels(frame: pd.DataFrame, i: int) -> dict[str, float]:
    """Get external liquidity levels for the current day."""
    levels = {}
    
    # Previous Day High/Low
    if "day_high" in frame.columns and "day_low" in frame.columns:
        levels["pdh"] = float(frame["day_high"].iat[i])
        levels["pdl"] = float(frame["day_low"].iat[i])
    
    # Previous Week High/Low
    if "week_high" in frame.columns and "week_low" in frame.columns:
        levels["pwh"] = float(frame["week_high"].iat[i])
        levels["pwl"] = float(frame["week_low"].iat[i])
    
    # Weekly Open
    if "week_open" in frame.columns:
        levels["weekly_open"] = float(frame["week_open"].iat[i])
    
    # Asia Session High/Low
    if "session_range_high_19_00_03_00" in frame.columns:
        levels["asia_high"] = float(frame["session_range_high_19_00_03_00"].iat[i])
        levels["asia_low"] = float(frame["session_range_low_19_00_03_00"].iat[i])
    
    # London Session High/Low
    if "session_range_high_03_00_07_00" in frame.columns:
        levels["london_high"] = float(frame["session_range_high_03_00_07_00"].iat[i])
        levels["london_low"] = float(frame["session_range_low_03_00_07_00"].iat[i])
    
    # Filter out NaN values
    return {k: v for k, v in levels.items() if v == v}


def _check_sweep(frame: pd.DataFrame, i: int, level: float, is_high: bool, params: dict[str, Any]) -> dict[str, Any] | None:
    """Check if a valid sweep occurred at index i."""
    if i < 1:
        return None
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    max_extension = atr * float(params["sweep_extension_atr"])
    
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    sweep_buffer = 1.0 * PIP_SIZE
    
    if is_high:
        # High sweep: price went above level but closed below
        if curr_high >= (level + sweep_buffer) and curr_close <= level:
            extension = curr_high - level
            if extension <= max_extension:
                return {
                    "direction": "short",
                    "sweep_extreme": curr_high,
                    "sweep_bar_low": curr_low,
                }
    else:
        # Low sweep: price went below level but closed above
        if curr_low <= (level - sweep_buffer) and curr_close >= level:
            extension = level - curr_low
            if extension <= max_extension:
                return {
                    "direction": "long",
                    "sweep_extreme": curr_low,
                    "sweep_bar_high": curr_high,
                }
    
    return None


def _check_displacement(frame: pd.DataFrame, i: int, direction: str, params: dict[str, Any]) -> bool:
    """Check if a valid displacement bar occurred at index i."""
    if i < 1:
        return False
    
    curr_open = float(frame["open"].iat[i])
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    body = abs(curr_close - curr_open)
    candle_range = curr_high - curr_low
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    
    # Displacement thresholds
    body_atr_ratio = body / atr if atr > 0 else 0
    body_fraction = body / candle_range if candle_range > 0 else 0
    close_location = (curr_close - curr_low) / candle_range if candle_range > 0 else 0.5
    
    min_body_atr = float(params["displacement_body_atr"])
    min_body_fraction = float(params["displacement_body_fraction"])
    
    if direction == "long":
        min_close_loc = float(params["displacement_close_location_long"])
        return (body_atr_ratio >= min_body_atr and 
                body_fraction >= min_body_fraction and 
                close_location >= min_close_loc and
                curr_close > curr_open)
    else:
        max_close_loc = float(params["displacement_close_location_short"])
        return (body_atr_ratio >= min_body_atr and 
                body_fraction >= min_body_fraction and 
                close_location <= max_close_loc and
                curr_close < curr_open)


def _get_last_swing_point(frame: pd.DataFrame, i: int, direction: str) -> float | None:
    """Get the last confirmed swing high or low before index i."""
    if "swing_high" not in frame.columns or "swing_low" not in frame.columns:
        return None
    
    lookback = 20
    for j in range(i - 1, max(0, i - lookback), -1):
        if direction == "long":
            swing_low = float(frame["swing_low"].iat[j])
            if swing_low == swing_low and swing_low > 0:
                return swing_low
        else:
            swing_high = float(frame["swing_high"].iat[j])
            if swing_high == swing_high and swing_high > 0:
                return swing_high
    
    return None


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    
    # Check if in trading window
    if not is_in_session(ts, str(params["session_name"])) or not _is_time_to_trade(ts, params):
        return None
    
    # Check if past force close time
    if _is_past_force_close(ts, params):
        return None
    
    # Need enough history
    if i < 20:
        return None
    
    # Get liquidity levels
    levels = _get_liquidity_levels(frame, i)
    if not levels:
        return None
    
    # Track if we already had a sweep today
    today = ts.date()
    sweep_today = None
    
    # Look for valid sweep in recent history
    lookback_bars = 24  # 2 hours in M5
    for j in range(i - 1, max(0, i - lookback_bars), -1):
        j_ts = frame.index[j]
        
        # Only check within trading window
        if not is_in_session(j_ts, str(params["session_name"])) or not _is_time_to_trade(j_ts, params):
            continue
        
        # Check each liquidity level for sweep
        for level_name, level_value in levels.items():
            if level_name in ["pdh", "pwh", "asia_high", "london_high"]:
                sweep = _check_sweep(frame, j, level_value, is_high=True, params=params)
                if sweep:
                    sweep_today = sweep
                    sweep_today["level_name"] = level_name
                    sweep_today["level_value"] = level_value
                    sweep_today["sweep_index"] = j
                    break
            elif level_name in ["pdl", "pwl", "weekly_open", "asia_low", "london_low"]:
                sweep = _check_sweep(frame, j, level_value, is_high=False, params=params)
                if sweep:
                    sweep_today = sweep
                    sweep_today["level_name"] = level_name
                    sweep_today["level_value"] = level_value
                    sweep_today["sweep_index"] = j
                    break
        
        if sweep_today:
            break
    
    if not sweep_today:
        return None
    
    # Now look for confirmation after the sweep
    confirmation_bars = int(params["confirmation_bars"])
    sweep_idx = sweep_today["sweep_index"]
    
    for k in range(sweep_idx + 1, min(i + 1, sweep_idx + confirmation_bars + 1)):
        if k >= len(frame):
            break
        
        # Check for displacement
        if _check_displacement(frame, k, sweep_today["direction"], params):
            # Additional confirmation: close beyond swing point
            last_swing = _get_last_swing_point(frame, k, sweep_today["direction"])
            
            # For long: close must be above last swing low and sweep bar high
            # For short: close must be below last swing high and sweep bar low
            curr_close = float(frame["close"].iat[k])
            
            if sweep_today["direction"] == "long":
                if last_swing and curr_close > last_swing and curr_close > sweep_today["sweep_bar_high"]:
                    # Valid long signal
                    sl = sweep_today["sweep_extreme"] - PIP_SIZE
                    entry_price = float(frame["open"].iat[min(k + 1, len(frame) - 1)])
                    
                    return {
                        "direction": "long",
                        "entry_mode": "market",
                        "stop_mode": "price",
                        "stop_price": sl,
                        "target_mode": "rr",
                        "target_rr": float(params["target_rr"]),
                        "max_hold_bars": int(params["max_hold_bars"]),
                        "cooldown_bars": int(params.get("cooldown_bars", 0)),
                        "break_even_at_r": params.get("break_even_at_r"),
                        "session_name": str(params["session_name"]),
                    }
            else:
                if last_swing and curr_close < last_swing and curr_close < sweep_today["sweep_bar_low"]:
                    # Valid short signal
                    sl = sweep_today["sweep_extreme"] + PIP_SIZE
                    entry_price = float(frame["open"].iat[min(k + 1, len(frame) - 1)])
                    
                    return {
                        "direction": "short",
                        "entry_mode": "market",
                        "stop_mode": "price",
                        "stop_price": sl,
                        "target_mode": "rr",
                        "target_rr": float(params["target_rr"]),
                        "max_hold_bars": int(params["max_hold_bars"]),
                        "cooldown_bars": int(params.get("cooldown_bars", 0)),
                        "break_even_at_r": params.get("break_even_at_r"),
                        "session_name": str(params["session_name"]),
                    }
    
    return None
