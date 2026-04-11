from __future__ import annotations

NAME = "trend_pullback"
WARMUP_BARS = 20


def parameter_grid(max_combinations: int = 8) -> list[dict]:
    combos = [
        {"use_h1_context": True, "rsi_low": 40, "rsi_high": 60, "pullback_atr": 0.30, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"use_h1_context": True, "rsi_low": 45, "rsi_high": 55, "pullback_atr": 0.25, "stop_atr": 1.0, "target_rr": 1.8, "break_even_enabled": False},
        {"use_h1_context": True, "rsi_low": 40, "rsi_high": 60, "pullback_atr": 0.40, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": True},
        {"use_h1_context": True, "rsi_low": 45, "rsi_high": 55, "pullback_atr": 0.35, "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": True},
        {"use_h1_context": False, "rsi_low": 40, "rsi_high": 60, "pullback_atr": 0.30, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"use_h1_context": False, "rsi_low": 45, "rsi_high": 55, "pullback_atr": 0.25, "stop_atr": 1.0, "target_rr": 1.8, "break_even_enabled": False},
        {"use_h1_context": False, "rsi_low": 40, "rsi_high": 60, "pullback_atr": 0.40, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": True},
        {"use_h1_context": False, "rsi_low": 45, "rsi_high": 55, "pullback_atr": 0.35, "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = frame["atr14"].iat[i]
    if atr <= 0:
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
    distance_atr = abs(frame["close"].iat[i] - frame["ema20"].iat[i]) / atr

    long_signal = (
        trend_up
        and distance_atr <= params["pullback_atr"]
        and frame["low"].iat[i] <= frame["ema20"].iat[i]
        and frame["rsi7"].iat[i] <= params["rsi_low"]
        and frame["close"].iat[i] > frame["high"].iat[i - 1]
    )
    short_signal = (
        trend_down
        and distance_atr <= params["pullback_atr"]
        and frame["high"].iat[i] >= frame["ema20"].iat[i]
        and frame["rsi7"].iat[i] >= params["rsi_high"]
        and frame["close"].iat[i] < frame["low"].iat[i - 1]
    )

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

