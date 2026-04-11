from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, day_range_ok, h1_trend_down, h1_trend_up


NAME = "donchian_breakout_m15"
WARMUP_BARS = 50


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"donchian_bars": 20, "use_h1_context": False, "breakout_candle_atr_max": 1.2, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"donchian_bars": 20, "use_h1_context": True, "breakout_candle_atr_max": 1.5, "stop_atr": 1.0, "target_rr": 2.0, "break_even_enabled": False},
        {"donchian_bars": 30, "use_h1_context": False, "breakout_candle_atr_max": 1.2, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": False},
        {"donchian_bars": 30, "use_h1_context": True, "breakout_candle_atr_max": 1.5, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    bars = params["donchian_bars"]
    high_level = float(frame[f"donchian_high_{bars}"].iat[i])
    low_level = float(frame[f"donchian_low_{bars}"].iat[i])
    if not (atr > 0 and day_range_ok(frame, i) and candle_is_not_noisy(frame, i, params["breakout_candle_atr_max"])):
        return None

    long_signal = float(frame["close"].iat[i]) > high_level and float(frame["prev_close"].iat[i]) <= high_level
    short_signal = float(frame["close"].iat[i]) < low_level and float(frame["prev_close"].iat[i]) >= low_level

    if params["use_h1_context"]:
        long_signal = long_signal and h1_trend_up(frame, i)
        short_signal = short_signal and h1_trend_down(frame, i)

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None
