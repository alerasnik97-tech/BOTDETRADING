from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import is_in_session


NAME = "campaign3_mtf_alignment"
WARMUP_BARS = 150
EXPLICIT_TIMEFRAME = "M5"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_11_1630",
        "session_name": "pm_11_16",
        "entry_start": "11:00",
        "entry_end": "16:30",
        "force_close": "16:30",
        "target_rr": 2.0,
        "max_hold_bars": 50,
        "cooldown_bars": 0,
        "break_even_at_r": 1.0,
        "ema_fast": 9,
        "ema_slow": 21,
        "pullback_min_atr": 0.2,
        "pullback_max_atr": 0.5,
        "displacement_body_atr": 0.20,
        "pullback_lookback_bars": 15,
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
    force_close = str(params.get("force_close", "16:30"))
    force_h, force_m = map(int, force_close.split(":"))
    force_total = force_h * 60 + force_m
    curr_total = ts.hour * 60 + ts.minute
    return curr_total >= force_total


def _calculate_ema(frame: pd.DataFrame, i: int, period: int) -> float | None:
    """Calculate EMA at index i."""
    if i < period:
        return None
    
    prices = [float(frame["close"].iat[j]) for j in range(i - period, i)]
    if not prices:
        return None
    
    # Simple EMA calculation
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def _check_htf_trend(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> str | None:
    """Check H1 trend alignment using EMAs."""
    # Use H1 columns if available, otherwise calculate from M5
    ema_fast_period = int(params["ema_fast"])
    ema_slow_period = int(params["ema_slow"])
    
    # Try to use H1 EMA columns if available
    if f"h1_ema{ema_fast_period}" in frame.columns and f"h1_ema{ema_slow_period}" in frame.columns:
        ema_fast = float(frame[f"h1_ema{ema_fast_period}"].iat[i])
        ema_slow = float(frame[f"h1_ema{ema_slow_period}"].iat[i])
        curr_price = float(frame["close"].iat[i])
        
        if ema_fast > ema_slow and curr_price > ema_fast:
            return "bullish"
        elif ema_fast < ema_slow and curr_price < ema_fast:
            return "bearish"
    
    return None


def _check_pullback(frame: pd.DataFrame, i: int, trend: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Check for pullback to EMA or value zone."""
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    min_pullback = atr * float(params["pullback_min_atr"])
    max_pullback = atr * float(params["pullback_max_atr"])
    
    ema_slow_period = int(params["ema_slow"])
    
    # Get EMA slow
    ema_slow = None
    if f"h1_ema{ema_slow_period}" in frame.columns:
        ema_slow = float(frame[f"h1_ema{ema_slow_period}"].iat[i])
    
    if ema_slow is None:
        return None
    
    curr_close = float(frame["close"].iat[i])
    
    if trend == "bullish":
        # Pullback to EMA from above
        pullback = ema_slow - curr_close
        if min_pullback <= pullback <= max_pullback and curr_close < ema_slow:
            return {"pullback_extreme": curr_close}
    else:
        # Pullback to EMA from below
        pullback = curr_close - ema_slow
        if min_pullback <= pullback <= max_pullback and curr_close > ema_slow:
            return {"pullback_extreme": curr_close}
    
    return None


def _check_displacement(frame: pd.DataFrame, i: int, trend: str, params: dict[str, Any]) -> bool:
    """Check for displacement bar in trend direction."""
    curr_open = float(frame["open"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    body = abs(curr_close - curr_open)
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    body_atr_ratio = body / atr if atr > 0 else 0
    
    min_body_atr = float(params["displacement_body_atr"])
    
    if body_atr_ratio < min_body_atr:
        return False
    
    if trend == "bullish":
        return curr_close > curr_open
    else:
        return curr_close < curr_open


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    
    # Check if in trading window
    if not is_in_session(ts, str(params["session_name"])) or not _is_time_to_trade(ts, params):
        return None
    
    # Check if past force close time
    if _is_past_force_close(ts, params):
        return None
    
    # Need enough history
    if i < 30:
        return None
    
    # Check HTF trend
    trend = _check_htf_trend(frame, i, params)
    if not trend:
        return None
    
    # Check for pullback
    pullback_info = _check_pullback(frame, i, trend, params)
    if not pullback_info:
        return None
    
    # Check for displacement bar in trend direction
    if not _check_displacement(frame, i, trend, params):
        return None
    
    # Valid MTF alignment signal
    sl = pullback_info["pullback_extreme"] - (2.0 * PIP_SIZE) if trend == "bullish" else pullback_info["pullback_extreme"] + (2.0 * PIP_SIZE)
    
    return {
        "direction": trend,
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
