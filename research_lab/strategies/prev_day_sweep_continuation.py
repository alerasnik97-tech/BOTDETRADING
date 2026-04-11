from __future__ import annotations

NAME = "prev_day_sweep_continuation"
WARMUP_BARS = 40


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"use_h1_context": False, "sweep_buffer_atr": 0.10, "stop_buffer_atr": 0.20, "target_rr": 1.5, "break_even_enabled": False},
        {"use_h1_context": True, "sweep_buffer_atr": 0.20, "stop_buffer_atr": 0.30, "target_rr": 2.0, "break_even_enabled": False},
        {"use_h1_context": False, "sweep_buffer_atr": 0.10, "stop_buffer_atr": 0.20, "target_rr": 1.5, "break_even_enabled": True},
        {"use_h1_context": True, "sweep_buffer_atr": 0.20, "stop_buffer_atr": 0.30, "target_rr": 2.0, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = frame["atr14"].iat[i]
    prev_day_high = frame["prev_day_high"].iat[i]
    prev_day_low = frame["prev_day_low"].iat[i]
    if not (atr > 0 and prev_day_high == prev_day_high and prev_day_low == prev_day_low and frame["range_atr"].iat[i] <= 1.8):
        return None

    use_h1 = params["use_h1_context"]
    long_bias = (not use_h1) or (frame["h1_ema50"].iat[i] > frame["h1_ema200"].iat[i] and frame["h1_ema200_slope_5"].iat[i] > 0)
    short_bias = (not use_h1) or (frame["h1_ema50"].iat[i] < frame["h1_ema200"].iat[i] and frame["h1_ema200_slope_5"].iat[i] < 0)
    sweep_buffer = atr * params["sweep_buffer_atr"]
    stop_buffer = atr * params["stop_buffer_atr"]

    long_signal = long_bias and frame["low"].iat[i] <= prev_day_low - sweep_buffer and frame["close"].iat[i] > prev_day_low and frame["close"].iat[i] > frame["open"].iat[i]
    short_signal = short_bias and frame["high"].iat[i] >= prev_day_high + sweep_buffer and frame["close"].iat[i] < prev_day_high and frame["close"].iat[i] < frame["open"].iat[i]

    if long_signal:
        return {"direction": "long", "stop_mode": "price", "stop_price": frame["low"].iat[i] - stop_buffer, "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "price", "stop_price": frame["high"].iat[i] + stop_buffer, "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

