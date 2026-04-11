from __future__ import annotations

NAME = "donchian_breakout"
WARMUP_BARS = 40


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"donchian_bars": 20, "use_h1_context": False, "breakout_candle_atr_max": 1.2, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"donchian_bars": 20, "use_h1_context": True, "breakout_candle_atr_max": 1.5, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": False},
        {"donchian_bars": 30, "use_h1_context": False, "breakout_candle_atr_max": 1.2, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": True},
        {"donchian_bars": 30, "use_h1_context": True, "breakout_candle_atr_max": 1.5, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    bars = params["donchian_bars"]
    donchian_high = frame[f"donchian_high_{bars}"].iat[i]
    donchian_low = frame[f"donchian_low_{bars}"].iat[i]
    if frame["range_atr"].iat[i] > params["breakout_candle_atr_max"]:
        return None

    use_h1 = params["use_h1_context"]
    long_bias = (not use_h1) or frame["h1_ema50"].iat[i] > frame["h1_ema200"].iat[i]
    short_bias = (not use_h1) or frame["h1_ema50"].iat[i] < frame["h1_ema200"].iat[i]

    long_signal = long_bias and frame["close"].iat[i] > donchian_high and frame["prev_close"].iat[i] <= donchian_high
    short_signal = short_bias and frame["close"].iat[i] < donchian_low and frame["prev_close"].iat[i] >= donchian_low
    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

