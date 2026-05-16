from __future__ import annotations

NAME = "compression_breakout"
WARMUP_BARS = 20


def parameter_grid(max_combinations: int = 8) -> list[dict]:
    combos = [
        {"use_h1_context": True, "donchian_bars": 20, "compression_atr_mult": 0.8, "breakout_candle_atr_max": 1.2, "stop_mode": "atr", "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"use_h1_context": True, "donchian_bars": 30, "compression_atr_mult": 1.0, "breakout_candle_atr_max": 1.5, "stop_mode": "atr", "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": False},
        {"use_h1_context": True, "donchian_bars": 30, "compression_atr_mult": 0.8, "breakout_candle_atr_max": 1.2, "stop_mode": "structure", "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": True},
        {"use_h1_context": True, "donchian_bars": 40, "compression_atr_mult": 1.0, "breakout_candle_atr_max": 1.5, "stop_mode": "structure", "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": True},
        {"use_h1_context": False, "donchian_bars": 20, "compression_atr_mult": 0.8, "breakout_candle_atr_max": 1.2, "stop_mode": "atr", "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"use_h1_context": False, "donchian_bars": 30, "compression_atr_mult": 1.0, "breakout_candle_atr_max": 1.5, "stop_mode": "atr", "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": False},
        {"use_h1_context": False, "donchian_bars": 30, "compression_atr_mult": 0.8, "breakout_candle_atr_max": 1.2, "stop_mode": "structure", "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": True},
        {"use_h1_context": False, "donchian_bars": 40, "compression_atr_mult": 1.0, "breakout_candle_atr_max": 1.5, "stop_mode": "structure", "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    bars = params["donchian_bars"]
    donchian_high = frame[f"donchian_high_{bars}"].iat[i]
    donchian_low = frame[f"donchian_low_{bars}"].iat[i]
    comp_range_atr = frame[f"donchian_range_atr_{bars}"].iat[i]
    atr = frame["atr14"].iat[i]
    if not (atr > 0 and comp_range_atr <= params["compression_atr_mult"] and frame["range_atr"].iat[i] <= params["breakout_candle_atr_max"]):
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

    long_signal = trend_up and frame["close"].iat[i] > donchian_high and frame["prev_close"].iat[i] <= donchian_high and frame["close"].iat[i] >= frame["ema20"].iat[i]
    short_signal = trend_down and frame["close"].iat[i] < donchian_low and frame["prev_close"].iat[i] >= donchian_low and frame["close"].iat[i] <= frame["ema20"].iat[i]

    if long_signal:
        signal = {"direction": "long", "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
        if params["stop_mode"] == "structure":
            signal.update({"stop_mode": "price", "stop_price": donchian_low})
        else:
            signal.update({"stop_mode": "atr", "stop_atr": params["stop_atr"]})
        return signal
    if short_signal:
        signal = {"direction": "short", "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
        if params["stop_mode"] == "structure":
            signal.update({"stop_mode": "price", "stop_price": donchian_high})
        else:
            signal.update({"stop_mode": "atr", "stop_atr": params["stop_atr"]})
        return signal
    return None
