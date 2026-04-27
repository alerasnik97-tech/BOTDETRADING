from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import is_in_session


NAME = "campaign3_midday_daily_reclaim"
WARMUP_BARS = 150
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_11_14",
        "session_name": "pm_11_16",
        "entry_start": "11:00",
        "entry_end": "14:00",
        "force_close": "14:00",
        "target_rr": 2.0,
        "max_hold_bars": 30,
        "cooldown_bars": 0,
        "break_even_at_r": 1.0,
        "false_break_pips": 5.0,
        "false_break_max_atr": 0.3,
        "false_break_bars": 3,
        "reclaim_bars": 8,
        "displacement_body_atr": 0.20,
        "displacement_body_fraction": 0.40,
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
    force_close = str(params.get("force_close", "14:00"))
    force_h, force_m = map(int, force_close.split(":"))
    force_total = force_h * 60 + force_m
    curr_total = ts.hour * 60 + ts.minute
    return curr_total >= force_total


def _get_daily_levels(frame: pd.DataFrame, i: int) -> dict[str, float]:
    """Get daily objective levels."""
    levels = {}
    
    # Previous Day High/Low
    if "day_high" in frame.columns and "day_low" in frame.columns:
        levels["pdh"] = float(frame["day_high"].iat[i])
        levels["pdl"] = float(frame["day_low"].iat[i])
    
    # Weekly Open
    if "week_open" in frame.columns:
        levels["weekly_open"] = float(frame["week_open"].iat[i])
    
    # Monthly Open (if within 50 pips)
    if "month_open" in frame.columns:
        month_open = float(frame["month_open"].iat[i])
        curr_price = float(frame["close"].iat[i])
        if abs(curr_price - month_open) <= 0.0050:  # 50 pips
            levels["monthly_open"] = month_open
    
    return {k: v for k, v in levels.items() if v == v}


def _check_false_break(frame: pd.DataFrame, i: int, level: float, is_high: bool, params: dict[str, Any]) -> dict[str, Any] | None:
    """Check if a false break occurred at index i."""
    if i < 1:
        return None
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    max_extension = atr * float(params["false_break_max_atr"])
    break_pips = float(params["false_break_pips"]) * PIP_SIZE
    
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    if is_high:
        # False break high: broke level by >= break_pips, closed back below
        if curr_high >= (level + break_pips):
            extension = curr_high - level
            if extension <= max_extension and curr_close < level:
                return {
                    "direction": "short",
                    "false_break_extreme": curr_high,
                }
    else:
        # False break low: broke level by >= break_pips, closed back above
        if curr_low <= (level - break_pips):
            extension = level - curr_low
            if extension <= max_extension and curr_close > level:
                return {
                    "direction": "long",
                    "false_break_extreme": curr_low,
                }
    
    return None


def _check_reclaim_confirmation(frame: pd.DataFrame, i: int, direction: str, level: float, params: dict[str, Any]) -> bool:
    """Check if reclaim was confirmed with displacement."""
    curr_open = float(frame["open"].iat[i])
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    body = abs(curr_close - curr_open)
    candle_range = curr_high - curr_low
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    
    body_atr_ratio = body / atr if atr > 0 else 0
    body_fraction = body / candle_range if candle_range > 0 else 0
    
    min_body_atr = float(params["displacement_body_atr"])
    min_body_fraction = float(params["displacement_body_fraction"])
    
    if not (body_atr_ratio >= min_body_atr and body_fraction >= min_body_fraction):
        return False
    
    if direction == "long":
        return curr_close > level and curr_close > curr_open
    else:
        return curr_close < level and curr_close < curr_open


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
    
    # Get daily levels
    levels = _get_daily_levels(frame, i)
    if not levels:
        return None
    
    # Look for false break in recent history
    false_break_today = None
    lookback_bars = 16  # 4 hours in M15
    
    for j in range(i - 1, max(0, i - lookback_bars), -1):
        j_ts = frame.index[j]
        
        # Only check within trading window
        if not is_in_session(j_ts, str(params["session_name"])) or not _is_time_to_trade(j_ts, params):
            continue
        
        # Check each level for false break
        for level_name, level_value in levels.items():
            if level_name in ["pdh", "weekly_open", "monthly_open"]:
                fb = _check_false_break(frame, j, level_value, is_high=True, params=params)
                if fb:
                    false_break_today = fb
                    false_break_today["level_name"] = level_name
                    false_break_today["level_value"] = level_value
                    false_break_today["false_break_index"] = j
                    break
            elif level_name in ["pdl", "weekly_open", "monthly_open"]:
                fb = _check_false_break(frame, j, level_value, is_high=False, params=params)
                if fb:
                    false_break_today = fb
                    false_break_today["level_name"] = level_name
                    false_break_today["level_value"] = level_value
                    false_break_today["false_break_index"] = j
                    break
        
        if false_break_today:
            break
    
    if not false_break_today:
        return None
    
    # Check for reclaim confirmation after false break
    reclaim_bars = int(params["reclaim_bars"])
    fb_idx = false_break_today["false_break_index"]
    
    for k in range(fb_idx + 1, min(i + 1, fb_idx + reclaim_bars + 1)):
        if k >= len(frame):
            break
        
        if _check_reclaim_confirmation(frame, k, false_break_today["direction"], false_break_today["level_value"], params):
            # Valid reclaim signal
            sl = false_break_today["false_break_extreme"] + (2.0 * PIP_SIZE if false_break_today["direction"] == "long" else -2.0 * PIP_SIZE)
            
            # Entry on next bar (M5)
            # Since we're in M15, we need to switch to M5 for entry
            # For now, entry on next M15 bar
            entry_price = float(frame["open"].iat[min(k + 1, len(frame) - 1)])
            
            return {
                "direction": false_break_today["direction"],
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
