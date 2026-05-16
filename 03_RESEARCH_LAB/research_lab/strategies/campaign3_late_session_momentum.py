from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import is_in_session


NAME = "campaign3_late_session_momentum"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M5"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "canonical_17_19",
        "session_name": "pm_1630_19",
        "entry_start": "17:00",
        "entry_end": "19:00",
        "force_close": "19:00",
        "target_rr": 1.5,
        "max_hold_bars": 15,
        "cooldown_bars": 0,
        "break_even_at_r": 0.8,
        "momentum_lookback_bars": 20,
        "momentum_threshold_bars": 12,
        "pullback_min_atr": 0.2,
        "pullback_max_atr": 0.4,
        "displacement_body_atr": 0.18,
        "min_atr": 0.0005,
        "max_spread_pips": 2.5,
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
    force_close = str(params.get("force_close", "19:00"))
    force_h, force_m = map(int, force_close.split(":"))
    force_total = force_h * 60 + force_m
    curr_total = ts.hour * 60 + ts.minute
    return curr_total >= force_total


def _determine_momentum(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> str | None:
    """Determine momentum direction."""
    lookback = int(params["momentum_lookback_bars"])
    threshold = int(params["momentum_threshold_bars"])
    
    if i < lookback:
        return None
    
    bullish_count = 0
    bearish_count = 0
    
    for j in range(i - lookback, i):
        curr_open = float(frame["open"].iat[j])
        curr_close = float(frame["close"].iat[j])
        
        if curr_close > curr_open:
            bullish_count += 1
        elif curr_close < curr_open:
            bearish_count += 1
    
    if bullish_count >= threshold:
        return "bullish"
    elif bearish_count >= threshold:
        return "bearish"
    
    return None


def _check_pullback(frame: pd.DataFrame, i: int, momentum: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Check if pullback occurred within ATR range."""
    if i < 1:
        return None
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    min_pullback = atr * float(params["pullback_min_atr"])
    max_pullback = atr * float(params["pullback_max_atr"])
    
    curr_low = float(frame["low"].iat[i])
    curr_high = float(frame["high"].iat[i])
    
    # Get recent high/low based on momentum
    lookback = 10
    if i < lookback:
        return None
    
    recent_high = max(float(frame["high"].iat[j]) for j in range(i - lookback, i))
    recent_low = min(float(frame["low"].iat[j]) for j in range(i - lookback, i))
    
    if momentum == "bullish":
        # Pullback against bullish momentum
        pullback = recent_high - curr_low
        if min_pullback <= pullback <= max_pullback:
            return {"pullback_extreme": curr_low}
    else:
        # Pullback against bearish momentum
        pullback = curr_high - recent_low
        if min_pullback <= pullback <= max_pullback:
            return {"pullback_extreme": curr_high}
    
    return None


def _check_displacement(frame: pd.DataFrame, i: int, momentum: str, params: dict[str, Any]) -> bool:
    """Check for displacement bar in momentum direction."""
    curr_open = float(frame["open"].iat[i])
    curr_close = float(frame["close"].iat[i])
    
    body = abs(curr_close - curr_open)
    
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    body_atr_ratio = body / atr if atr > 0 else 0
    
    min_body_atr = float(params["displacement_body_atr"])
    
    if body_atr_ratio < min_body_atr:
        return False
    
    if momentum == "bullish":
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
    if i < 25:
        return None
    
    # Volatility filter
    atr = float(frame["atr_14"].iat[i]) if "atr_14" in frame.columns else 0.0003
    min_atr = float(params["min_atr"])
    if atr < min_atr:
        return None
    
    # Spread filter (if available)
    if "spread" in frame.columns:
        spread = float(frame["spread"].iat[i])
        max_spread = float(params["max_spread_pips"]) * PIP_SIZE
        if spread > max_spread:
            return None
    
    # Determine momentum
    momentum = _determine_momentum(frame, i, params)
    if not momentum:
        return None
    
    # Check for pullback
    pullback_info = _check_pullback(frame, i, momentum, params)
    if not pullback_info:
        return None
    
    # Check for displacement bar in momentum direction
    if not _check_displacement(frame, i, momentum, params):
        return None
    
    # Valid momentum signal
    if momentum == "bullish":
        sl = pullback_info["pullback_extreme"] - (2.0 * PIP_SIZE)
    else:
        sl = pullback_info["pullback_extreme"] + (2.0 * PIP_SIZE)
    
    return {
        "direction": "long" if momentum == "bullish" else "short",
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
