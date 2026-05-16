from __future__ import annotations

NAME = "session_range_breakout"
WARMUP_BARS = 20


def parameter_grid(max_combinations: int = 8) -> list[dict]:
    combos = [
        {"range_end": "12:00", "use_h1_context": False, "breakout_candle_atr_max": 1.2, "target_rr": 1.5, "buffer_atr": 0.1, "break_even_enabled": False},
        {"range_end": "12:00", "use_h1_context": True, "breakout_candle_atr_max": 1.5, "target_rr": 1.8, "buffer_atr": 0.2, "break_even_enabled": False},
        {"range_end": "13:00", "use_h1_context": False, "breakout_candle_atr_max": 1.2, "target_rr": 1.5, "buffer_atr": 0.1, "break_even_enabled": True},
        {"range_end": "13:00", "use_h1_context": True, "breakout_candle_atr_max": 1.5, "target_rr": 1.8, "buffer_atr": 0.2, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    suffix = params["range_end"].replace(":", "_")
    range_high = frame[f"session_range_high_{suffix}"].iat[i]
    range_low = frame[f"session_range_low_{suffix}"].iat[i]
    complete = bool(frame[f"session_range_complete_{suffix}"].iat[i])
    atr = frame["atr14"].iat[i]
    if not (complete and atr > 0 and frame["range_atr"].iat[i] <= params["breakout_candle_atr_max"]):
        return None

    use_h1 = params["use_h1_context"]
    trend_up = (not use_h1) or (
        frame["h1_ema50"].iat[i] > frame["h1_ema200"].iat[i]
        and frame["h1_ema200_slope_5"].iat[i] > 0
        and frame["h1_adx14"].iat[i] >= 18
    )
    trend_down = (not use_h1) or (
        frame["h1_ema50"].iat[i] < frame["h1_ema200"].iat[i]
        and frame["h1_ema200_slope_5"].iat[i] < 0
        and frame["h1_adx14"].iat[i] >= 18
    )
    buffer = atr * params["buffer_atr"]

    long_signal = trend_up and frame["close"].iat[i] > range_high + buffer and frame["prev_close"].iat[i] <= range_high + buffer
    short_signal = trend_down and frame["close"].iat[i] < range_low - buffer and frame["prev_close"].iat[i] >= range_low - buffer
    if long_signal:
        return {"direction": "long", "stop_mode": "price", "stop_price": range_low, "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "price", "stop_price": range_high, "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

