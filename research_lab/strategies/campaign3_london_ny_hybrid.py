from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import is_in_session


NAME = "campaign3_london_ny_hybrid"
WARMUP_BARS = 150
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_03_11",
        "session_name": "london_ny",
        "entry_start": "03:00",
        "entry_end": "11:00",
        "force_close": "11:00",
        "target_rr": 2.2,
        "max_hold_bars": 35,
        "cooldown_bars": 0,
        "break_even_at_r": 1.0,
        "london_end": "07:00",
        "ny_start": "07:00",
        "london_break_pips": 5.0,
        "reclaim_bars": 2,
        "breakout_bars": 10,
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
    force_close = str(params.get("force_close", "11:00"))
    force_h, force_m = map(int, force_close.split(":"))
    force_total = force_h * 60 + force_m
    curr_total = ts.hour * 60 + ts.minute
    return curr_total >= force_total


def _is_london_session(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    """Check if timestamp is in London session (03:00-07:00 NY)."""
    minute_value = ts.hour * 60 + ts.minute
    start_h, start_m = map(int, str(params["london_end"]).split(":"))
    end_h, end_m = map(int, str(params["ny_start"]).split(":"))
    return (start_h * 60 + start_m) <= minute_value < (end_h * 60 + end_m)


def _get_london_levels(frame: pd.DataFrame, i: int) -> dict[str, float] | None:
    """Get London session high/low at session end."""
    if "session_range_high_03_00_07_00" in frame.columns and "session_range_low_03_00_07_00" in frame.columns:
        london_high = float(frame["session_range_high_03_00_07_00"].iat[i])
        london_low = float(frame["session_range_low_03_00_07_00"].iat[i])
        if london_high == london_high and london_low == london_low:
            return {"london_high": london_high, "london_low": london_low}
    return None


def _check_ny_open_break(frame: pd.DataFrame, i: int, london_levels: dict[str, float], params: dict[str, Any]) -> dict[str, Any] | None:
    """Check if NY open broke London levels."""
    break_pips = float(params["london_break_pips"]) * PIP_SIZE
    
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    london_high = london_levels["london_high"]
    london_low = london_levels["london_low"]
    
    # NY open broke London high
    if curr_high >= (london_high + break_pips):
        return {"direction": "short", "broken_level": london_high}
    
    # NY open broke London low
    if curr_low <= (london_low - break_pips):
        return {"direction": "long", "broken_level": london_low}
    
    return None


def _check_london_reclaim(frame: pd.DataFrame, i: int, break_info: dict[str, Any], reclaim_bars: int) -> bool:
    """Check if London level was reclaimed."""
    broken_level = break_info["broken_level"]
    
    for j in range(i, min(len(frame), i + reclaim_bars)):
        curr_close = float(frame["close"].iat[j])
        
        if break_info["direction"] == "short":
            # Reclaim of London high (bearish break, reclaim means close above)
            if curr_close > broken_level:
                return True
        else:
            # Reclaim of London low (bullish break, reclaim means close below)
            if curr_close < broken_level:
                return True
    
    return False


def _check_london_breakout(frame: pd.DataFrame, i: int, london_levels: dict[str, float], breakout_bars: int) -> dict[str, Any] | None:
    """Check for breakout of London range after 08:00."""
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    london_high = london_levels["london_high"]
    london_low = london_levels["london_low"]
    
    # Upward breakout
    if curr_close > london_high:
        return {"direction": "long", "breakout_extreme": curr_high}
    
    # Downward breakout
    if curr_close < london_low:
        return {"direction": "short", "breakout_extreme": curr_low}
    
    return None


def _check_displacement(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> bool:
    """Check for displacement bar."""
    curr_open = float(frame["open"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    body = abs(curr_close - curr_open)
    candle_range = float(frame["high"].iat[i]) - float(frame["low"].iat[i])
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    body_atr_ratio = body / atr if atr > 0 else 0
    body_fraction = body / candle_range if candle_range > 0 else 0
    
    min_body_atr = float(params["displacement_body_atr"])
    min_body_fraction = float(params["displacement_body_fraction"])
    
    return body_atr_ratio >= min_body_atr and body_fraction >= min_body_fraction


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
    
    # Get London levels
    london_levels = _get_london_levels(frame, i)
    if not london_levels:
        return None
    
    # Strategy A: London sweep + NY reclaim (03:00-08:00)
    if _is_london_session(ts, params):
        break_info = _check_ny_open_break(frame, i, london_levels, params)
        if break_info:
            reclaim_bars = int(params["reclaim_bars"])
            if _check_london_reclaim(frame, i, break_info, reclaim_bars):
                # Check for displacement after reclaim
                if _check_displacement(frame, i, params):
                    sl = london_levels["london_high"] + (2.0 * PIP_SIZE) if break_info["direction"] == "short" else london_levels["london_low"] - (2.0 * PIP_SIZE)
                    
                    return {
                        "direction": break_info["direction"],
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
    
    # Strategy B: London range breakout after 08:00
    else:
        breakout_bars = int(params["breakout_bars"])
        breakout = _check_london_breakout(frame, i, london_levels, breakout_bars)
        if breakout and _check_displacement(frame, i, params):
            sl = london_levels["london_low"] - (2.0 * PIP_SIZE) if breakout["direction"] == "long" else london_levels["london_high"] + (2.0 * PIP_SIZE)
            
            return {
                "direction": breakout["direction"],
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
