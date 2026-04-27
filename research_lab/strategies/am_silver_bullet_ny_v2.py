from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import finite, is_in_session


NAME = "am_silver_bullet_ny_v2"
WARMUP_BARS = 40
EXPLICIT_TIMEFRAME = "M1"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "m1_mss_midpoint_reprice",
        "session_name": "am_08_11",
        "entry_start": "10:00",
        "entry_end": "11:00",
        "target_rr": 2.0,
        "max_hold_bars": 60,
        "break_even_at_r": 1.0,
        "cooldown_bars": 0,
        "max_fvg_after_mss_bars": 3,
        "retest_window_bars": 10,
        "min_fvg_pips": 0.5,
        "stop_buffer_pips": 2.0,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _minute_value(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _is_trade_window(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    start_hour, start_minute = (int(part) for part in str(params["entry_start"]).split(":"))
    end_hour, end_minute = (int(part) for part in str(params["entry_end"]).split(":"))
    start_value = start_hour * 60 + start_minute
    end_value = end_hour * 60 + end_minute
    minute_value = _minute_value(ts)
    return start_value <= minute_value < end_value


def _midpoint_not_touched_yet(frame: pd.DataFrame, *, start_i: int, end_i: int, direction: str, price: float) -> bool:
    if end_i < start_i:
        return True
    window = frame.iloc[start_i : end_i + 1]
    if window.empty:
        return True
    if direction == "long":
        return float(window["low"].min()) > price
    return float(window["high"].max()) < price


def _build_signal(
    *,
    direction: str,
    entry_price: float,
    stop_price: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "direction": direction,
        "entry_mode": "limit",
        "limit_price": entry_price,
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "rr",
        "target_rr": float(params["target_rr"]),
        "max_hold_bars": int(params["max_hold_bars"]),
        "cooldown_bars": int(params.get("cooldown_bars", 0)),
        "break_even_at_r": params.get("break_even_at_r"),
        "session_name": str(params["session_name"]),
    }


def _find_recent_setup(frame: pd.DataFrame, i: int, params: dict[str, Any], *, direction: str) -> dict[str, float] | None:
    fvg_col = "bullish_fvg" if direction == "long" else "bearish_fvg"
    fvg_mid_col = "bullish_fvg_mid" if direction == "long" else "bearish_fvg_mid"
    fvg_size_col = "bullish_fvg_size_pips" if direction == "long" else "bearish_fvg_size_pips"
    choch_col = "bullish_choch" if direction == "long" else "bearish_choch"
    anchor_col = "ctx_m5_sb_anchor_low" if direction == "long" else "ctx_m5_sb_anchor_high"
    sweep_col = "ctx_m5_swept_anchor_low" if direction == "long" else "ctx_m5_swept_anchor_high"

    search_start = max(0, i - int(params["retest_window_bars"]) - int(params["max_fvg_after_mss_bars"]))
    for k in range(i, search_start - 1, -1):
        if not _is_trade_window(frame.index[k], params):
            continue
        if not bool(frame[fvg_col].iat[k]):
            continue

        midpoint = float(frame[fvg_mid_col].iat[k])
        fvg_size = float(frame[fvg_size_col].iat[k])
        anchor_level = float(frame[anchor_col].iat[k])
        sweep_active = bool(frame[sweep_col].iat[k])
        if not finite(midpoint) or not finite(fvg_size) or not finite(anchor_level) or not sweep_active:
            continue
        if fvg_size < float(params["min_fvg_pips"]):
            continue
        if (i - k) > int(params["retest_window_bars"]):
            continue

        choch_start = max(0, k - int(params["max_fvg_after_mss_bars"]))
        has_recent_choch = any(bool(frame[choch_col].iat[j]) for j in range(k, choch_start - 1, -1))
        if not has_recent_choch:
            continue

        if not _midpoint_not_touched_yet(frame, start_i=k + 1, end_i=i, direction=direction, price=midpoint):
            continue

        stop_buffer = float(params["stop_buffer_pips"]) * PIP_SIZE
        stop_price = anchor_level - stop_buffer if direction == "long" else anchor_level + stop_buffer
        if direction == "long" and midpoint <= stop_price:
            continue
        if direction == "short" and midpoint >= stop_price:
            continue
        return {"entry_price": midpoint, "stop_price": stop_price}
    return None


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 3 or not is_in_session(ts, str(params["session_name"])) or not _is_trade_window(ts, params):
        return None

    long_setup = _find_recent_setup(frame, i, params, direction="long")
    if long_setup is not None:
        return _build_signal(direction="long", params=params, **long_setup)

    short_setup = _find_recent_setup(frame, i, params, direction="short")
    if short_setup is not None:
        return _build_signal(direction="short", params=params, **short_setup)

    return None

generate_signal = signal
