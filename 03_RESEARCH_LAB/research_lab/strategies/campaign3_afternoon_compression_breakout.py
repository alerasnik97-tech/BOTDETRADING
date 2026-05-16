from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import is_in_session


NAME = "campaign3_afternoon_compression_breakout"
WARMUP_BARS = 150
EXPLICIT_TIMEFRAME = "M15"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_14_17",
        "session_name": "pm_11_17",
        "entry_start": "14:00",
        "entry_end": "17:00",
        "force_close": "17:00",
        "target_rr": 2.5,
        "max_hold_bars": 40,
        "cooldown_bars": 0,
        "break_even_at_r": 1.0,
        "compression_bars": 8,
        "compression_max_atr": 0.4,
        "breakout_pips": 3.0,
        "fakeout_bars": 3,
        "displacement_body_atr": 0.25,
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
    force_close = str(params.get("force_close", "17:00"))
    force_h, force_m = map(int, force_close.split(":"))
    force_total = force_h * 60 + force_m
    curr_total = ts.hour * 60 + ts.minute
    return curr_total >= force_total


def _check_compression(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    """Check if compression pattern exists."""
    compression_bars = int(params["compression_bars"])
    
    if i < compression_bars:
        return None
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    max_range = atr * float(params["compression_max_atr"])
    
    # Calculate range of last N bars
    highs = [float(frame["high"].iat[j]) for j in range(i - compression_bars, i)]
    lows = [float(frame["low"].iat[j]) for j in range(i - compression_bars, i)]
    
    range_size = max(highs) - min(lows)
    
    if range_size <= max_range:
        return {
            "compression_high": max(highs),
            "compression_low": min(lows),
        }
    
    return None


def _check_breakout(frame: pd.DataFrame, i: int, compression: dict[str, float], params: dict[str, Any]) -> dict[str, Any] | None:
    """Check if valid breakout occurred."""
    breakout_pips = float(params["breakout_pips"]) * PIP_SIZE
    
    curr_high = float(frame["high"].iat[i])
    curr_low = float(frame["low"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    comp_high = compression["compression_high"]
    comp_low = compression["compression_low"]
    
    # Check for upward breakout
    if curr_high >= (comp_high + breakout_pips) and curr_close > comp_high:
        return {
            "direction": "long",
            "breakout_extreme": curr_high,
        }
    
    # Check for downward breakout
    if curr_low <= (comp_low - breakout_pips) and curr_close < comp_low:
        return {
            "direction": "short",
            "breakout_extreme": curr_low,
        }
    
    return None


def _check_fakeout(frame: pd.DataFrame, i: int, compression: dict[str, float], fakeout_bars: int) -> bool:
    """Check if price re-entered compression range (fakeout)."""
    comp_high = compression["compression_high"]
    comp_low = compression["compression_low"]
    
    for j in range(i + 1, min(len(frame), i + fakeout_bars + 1)):
        curr_close = float(frame["close"].iat[j])
        if comp_low <= curr_close <= comp_high:
            return True
    
    return False


def _check_displacement(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> bool:
    """Check for displacement bar."""
    curr_open = float(frame["open"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    body = abs(curr_close - curr_open)
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    body_atr_ratio = body / atr if atr > 0 else 0
    
    min_body_atr = float(params["displacement_body_atr"])
    
    return body_atr_ratio >= min_body_atr


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
    
    # Check for compression
    compression = _check_compression(frame, i, params)
    if not compression:
        return None
    
    # Check for breakout
    breakout = _check_breakout(frame, i, compression, params)
    if not breakout:
        return None
    
    # Check for displacement
    if not _check_displacement(frame, i, params):
        return None
    
    # Check for fakeout in next few bars
    fakeout_bars = int(params["fakeout_bars"])
    if _check_fakeout(frame, i, compression, fakeout_bars):
        return None
    
    # Valid breakout signal
    if breakout["direction"] == "long":
        sl = compression["compression_low"] - (2.0 * PIP_SIZE)
    else:
        sl = compression["compression_high"] + (2.0 * PIP_SIZE)
    
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
