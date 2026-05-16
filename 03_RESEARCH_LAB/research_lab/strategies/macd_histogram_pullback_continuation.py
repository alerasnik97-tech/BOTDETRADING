from __future__ import annotations

import math

from research_lab.strategies.common import stratified_sample_combinations


NAME = "macd_histogram_pullback_continuation"
WARMUP_BARS = 120


def parameter_space() -> dict[str, list]:
    return {
        "ema_fast": [20, 30],
        "ema_slow": [50, 100],
        "adx_min": [18, 20, 22],
        "macd_fast": [8, 12],
        "macd_slow": [17, 26],
        "macd_signal": [5, 9],
        "pullback_filter": ["ema_fast_touch", "close_below_above_ema_fast", "atr_pullback"],
        "atr_stop": [1.0, 1.2, 1.5],
        "target_rr": [1.2, 1.5, 2.0],
        "trailing_mode": ["off", "atr"],
        "break_even_at_r": [None, 1.0],
        "session_name": ["light_fixed"],
    }


def parameter_grid(max_combinations: int = 12, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def _pullback_passes(frame, i: int, params: dict, direction: str, ema_fast_value: float, ema_fast_prev: float, atr: float) -> bool:
    close = float(frame["close"].iat[i])
    prev_close = float(frame["prev_close"].iat[i])
    low = float(frame["low"].iat[i])
    high = float(frame["high"].iat[i])
    mode = str(params["pullback_filter"])
    tolerance = atr * 0.25

    if direction == "long":
        if mode == "ema_fast_touch":
            return low <= ema_fast_value and close >= ema_fast_value
        if mode == "close_below_above_ema_fast":
            return prev_close <= ema_fast_prev and close > ema_fast_value
        return abs(low - ema_fast_value) <= tolerance and close > ema_fast_value

    if mode == "ema_fast_touch":
        return high >= ema_fast_value and close <= ema_fast_value
    if mode == "close_below_above_ema_fast":
        return prev_close >= ema_fast_prev and close < ema_fast_value
    return abs(high - ema_fast_value) <= tolerance and close < ema_fast_value


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    adx_value = float(frame["adx14"].iat[i])
    if not (math.isfinite(atr) and atr > 0 and math.isfinite(adx_value) and adx_value >= float(params["adx_min"])):
        return None

    ema_fast_col = f"ema{params['ema_fast']}"
    ema_slow_col = f"ema{params['ema_slow']}"
    ema_fast_value = float(frame[ema_fast_col].iat[i])
    ema_fast_prev = float(frame[ema_fast_col].iat[i - 1])
    ema_slow_value = float(frame[ema_slow_col].iat[i])
    ema_slow_prev = float(frame[ema_slow_col].iat[i - 1])
    macd_hist_col = f"macd_hist_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"
    hist_now = float(frame[macd_hist_col].iat[i])
    hist_prev = float(frame[macd_hist_col].iat[i - 1])
    if not all(math.isfinite(value) for value in (ema_fast_value, ema_fast_prev, ema_slow_value, ema_slow_prev, hist_now, hist_prev)):
        return None

    near_zero_prev = abs(hist_prev) <= atr * 0.15
    long_bias = ema_fast_value > ema_slow_value and ema_fast_value > ema_fast_prev and ema_slow_value >= ema_slow_prev
    short_bias = ema_fast_value < ema_slow_value and ema_fast_value < ema_fast_prev and ema_slow_value <= ema_slow_prev

    long_signal = (
        long_bias
        and near_zero_prev
        and hist_prev <= 0.0
        and hist_now > 0.0
        and _pullback_passes(frame, i, params, "long", ema_fast_value, ema_fast_prev, atr)
    )
    short_signal = (
        short_bias
        and near_zero_prev
        and hist_prev >= 0.0
        and hist_now < 0.0
        and _pullback_passes(frame, i, params, "short", ema_fast_value, ema_fast_prev, atr)
    )

    if long_signal:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["atr_stop"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": params["trailing_mode"] == "atr",
            "session_name": params["session_name"],
        }

    if short_signal:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["atr_stop"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": params["trailing_mode"] == "atr",
            "session_name": params["session_name"],
        }

    return None
