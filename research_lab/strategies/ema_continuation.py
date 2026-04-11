from __future__ import annotations

NAME = "ema_continuation"
WARMUP_BARS = 30


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"rsi_low": 42, "rsi_high": 58, "pullback_atr": 0.25, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"rsi_low": 45, "rsi_high": 55, "pullback_atr": 0.40, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": False},
        {"rsi_low": 42, "rsi_high": 58, "pullback_atr": 0.25, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": True},
        {"rsi_low": 45, "rsi_high": 55, "pullback_atr": 0.40, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = frame["atr14"].iat[i]
    if atr <= 0 or frame["range_atr"].iat[i] > 1.5:
        return None

    long_bias = frame["h1_ema50"].iat[i] > frame["h1_ema200"].iat[i] and frame["h1_ema200_slope_5"].iat[i] > 0
    short_bias = frame["h1_ema50"].iat[i] < frame["h1_ema200"].iat[i] and frame["h1_ema200_slope_5"].iat[i] < 0
    distance_atr = abs(frame["close"].iat[i] - frame["ema20"].iat[i]) / atr

    long_signal = long_bias and distance_atr <= params["pullback_atr"] and frame["low"].iat[i] <= frame["ema20"].iat[i] and frame["rsi7"].iat[i] <= params["rsi_low"] and frame["close"].iat[i] > frame["open"].iat[i] and frame["close"].iat[i] > frame["prev_close"].iat[i]
    short_signal = short_bias and distance_atr <= params["pullback_atr"] and frame["high"].iat[i] >= frame["ema20"].iat[i] and frame["rsi7"].iat[i] >= params["rsi_high"] and frame["close"].iat[i] < frame["open"].iat[i] and frame["close"].iat[i] < frame["prev_close"].iat[i]

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

