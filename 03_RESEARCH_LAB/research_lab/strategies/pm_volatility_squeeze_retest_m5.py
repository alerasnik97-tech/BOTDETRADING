from __future__ import annotations

import math

import pandas as pd


NAME = "pm_volatility_squeeze_retest_m5"
WARMUP_BARS = 120
EXPLICIT_TIMEFRAME = "M5"

PM_TRIGGER_START_MINUTE = 13 * 60
PM_TRIGGER_END_MINUTE = 16 * 60
PIP = 0.0001


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {
            "variant_label": "base_tight",
            "bb_std": 2.0,
            "kc_mult": 1.5,
            "min_squeeze_bars": 5,
            "breakout_buffer_pips": 0.4,
            "limit_expiry_bars": 4,
            "tp_atr_mult": 0.8,
            "be_at_r": None,
            "max_hold_bars": 8,
            "body_pct_filter": 0.55,
            "expansion_atr_min": 0.9,
        },
        {
            "variant_label": "base_break_even",
            "bb_std": 2.0,
            "kc_mult": 1.5,
            "min_squeeze_bars": 5,
            "breakout_buffer_pips": 0.5,
            "limit_expiry_bars": 5,
            "tp_atr_mult": 0.9,
            "be_at_r": 0.75,
            "max_hold_bars": 6,
            "body_pct_filter": 0.55,
            "expansion_atr_min": 1.0,
        },
        {
            "variant_label": "wide_channel",
            "bb_std": 2.0,
            "kc_mult": 2.0,
            "min_squeeze_bars": 4,
            "breakout_buffer_pips": 0.3,
            "limit_expiry_bars": 5,
            "tp_atr_mult": 1.0,
            "be_at_r": None,
            "max_hold_bars": 8,
            "body_pct_filter": 0.50,
            "expansion_atr_min": 1.0,
        },
        {
            "variant_label": "bb22_strict",
            "bb_std": 2.2,
            "kc_mult": 1.5,
            "min_squeeze_bars": 5,
            "breakout_buffer_pips": 0.5,
            "limit_expiry_bars": 4,
            "tp_atr_mult": 0.8,
            "be_at_r": None,
            "max_hold_bars": 6,
            "body_pct_filter": 0.60,
            "expansion_atr_min": 1.1,
        },
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def _bb_suffix(std: float) -> str:
    return "20_2_2" if math.isclose(float(std), 2.2, rel_tol=0.0, abs_tol=1e-9) else "20_2_0"


def _kc_suffix(mult: float) -> str:
    return f"20_{str(float(mult)).replace('.', '_')}"


def _in_trigger_window(ts: pd.Timestamp) -> bool:
    minute_value = ts.hour * 60 + ts.minute
    return PM_TRIGGER_START_MINUTE <= minute_value < PM_TRIGGER_END_MINUTE


def _squeeze_slice(frame: pd.DataFrame, start_idx: int, end_idx: int, bb_upper_col: str, bb_lower_col: str, kc_upper_col: str, kc_lower_col: str) -> bool:
    bb_upper = frame[bb_upper_col].iloc[start_idx : end_idx + 1]
    bb_lower = frame[bb_lower_col].iloc[start_idx : end_idx + 1]
    kc_upper = frame[kc_upper_col].iloc[start_idx : end_idx + 1]
    kc_lower = frame[kc_lower_col].iloc[start_idx : end_idx + 1]
    return bool(((bb_upper <= kc_upper) & (bb_lower >= kc_lower)).all())


def signal(frame: pd.DataFrame, i: int, params: dict) -> dict | None:
    if not _in_trigger_window(frame.index[i]):
        return None

    bb_suffix = _bb_suffix(float(params["bb_std"]))
    kc_suffix = _kc_suffix(float(params["kc_mult"]))
    bb_upper_col = f"bb_upper_{bb_suffix}"
    bb_lower_col = f"bb_lower_{bb_suffix}"
    kc_upper_col = f"kc_upper_{kc_suffix}"
    kc_lower_col = f"kc_lower_{kc_suffix}"

    min_squeeze_bars = int(params["min_squeeze_bars"])
    limit_expiry_bars = int(params["limit_expiry_bars"])
    breakout_buffer = float(params["breakout_buffer_pips"]) * PIP
    body_pct_filter = float(params["body_pct_filter"])
    expansion_atr_min = float(params["expansion_atr_min"])
    tp_atr_mult = float(params["tp_atr_mult"])

    high_i = float(frame["high"].iat[i])
    low_i = float(frame["low"].iat[i])
    close_i = float(frame["close"].iat[i])

    for lookback in range(1, limit_expiry_bars + 1):
        j = i - lookback
        squeeze_start = j - min_squeeze_bars + 1
        if j <= 1 or squeeze_start < 1:
            continue

        if not _squeeze_slice(frame, squeeze_start, j, bb_upper_col, bb_lower_col, kc_upper_col, kc_lower_col):
            continue

        bb_upper_j = float(frame[bb_upper_col].iat[j])
        bb_lower_j = float(frame[bb_lower_col].iat[j])
        bb_upper_prev = float(frame[bb_upper_col].iat[j - 1])
        bb_lower_prev = float(frame[bb_lower_col].iat[j - 1])
        close_prev = float(frame["close"].iat[j - 1])
        open_j = float(frame["open"].iat[j])
        high_j = float(frame["high"].iat[j])
        low_j = float(frame["low"].iat[j])
        close_j = float(frame["close"].iat[j])
        atr_j = float(frame["atr14"].iat[j])
        range_atr_j = float(frame["range_atr"].iat[j])

        if atr_j <= 0 or range_atr_j < expansion_atr_min:
            continue

        bar_range = high_j - low_j
        if bar_range <= 0:
            continue

        body_pct = abs(close_j - open_j) / bar_range
        if body_pct < body_pct_filter:
            continue

        is_break_long = close_prev <= bb_upper_prev + breakout_buffer and close_j > bb_upper_j + breakout_buffer
        is_break_short = close_prev >= bb_lower_prev - breakout_buffer and close_j < bb_lower_j - breakout_buffer
        if not (is_break_long or is_break_short):
            continue

        target_dist = tp_atr_mult * atr_j
        if target_dist <= 0:
            continue

        between = frame.iloc[j + 1 : i]
        if is_break_long:
            if not between.empty:
                if bool((between["high"] >= bb_upper_j + target_dist).any()):
                    continue
                if bool((between["close"] < bb_lower_j).any()):
                    continue
            if low_i > bb_upper_j:
                continue
            target_price = bb_upper_j + target_dist
            if bb_lower_j >= bb_upper_j or target_price <= close_i or bb_lower_j >= close_i:
                continue
            return {
                "direction": "long",
                "stop_mode": "price",
                "stop_price": bb_lower_j,
                "target_mode": "price",
                "target_price": target_price,
                "target_rr": 0.0,
                "max_hold_bars": int(params["max_hold_bars"]),
                "break_even_at_r": params.get("be_at_r"),
                "trailing_atr": False,
                "session_name": "light_fixed",
                "signal_price": bb_upper_j,
            }

        if not between.empty:
            if bool((between["low"] <= bb_lower_j - target_dist).any()):
                continue
            if bool((between["close"] > bb_upper_j).any()):
                continue
        if high_i < bb_lower_j:
            continue
        target_price = bb_lower_j - target_dist
        if bb_upper_j <= bb_lower_j or target_price >= close_i or bb_upper_j <= close_i:
            continue
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": bb_upper_j,
            "target_mode": "price",
            "target_price": target_price,
            "target_rr": 0.0,
            "max_hold_bars": int(params["max_hold_bars"]),
            "break_even_at_r": params.get("be_at_r"),
            "trailing_atr": False,
            "session_name": "light_fixed",
            "signal_price": bb_lower_j,
        }

    return None
