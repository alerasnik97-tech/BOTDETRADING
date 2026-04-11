from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, day_range_ok, h1_trend_down, h1_trend_up


NAME = "impulse_pullback"
WARMUP_BARS = 80


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"impulse_atr": 0.8, "pullback_atr": 0.35, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"impulse_atr": 1.0, "pullback_atr": 0.35, "stop_atr": 1.0, "target_rr": 1.8, "break_even_enabled": False},
        {"impulse_atr": 0.8, "pullback_atr": 0.50, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": False},
        {"impulse_atr": 1.0, "pullback_atr": 0.50, "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (atr > 0 and candle_is_not_noisy(frame, i, 1.5) and day_range_ok(frame, i)):
        return None

    distance_atr = abs(float(frame["close"].iat[i]) - float(frame["ema20"].iat[i])) / atr
    long_impulse = float(frame["m15_close_change_atr_2"].iat[i]) >= params["impulse_atr"] and float(frame["m15_close"].iat[i]) >= float(frame["m15_ema20"].iat[i])
    short_impulse = float(frame["m15_close_change_atr_2"].iat[i]) <= -params["impulse_atr"] and float(frame["m15_close"].iat[i]) <= float(frame["m15_ema20"].iat[i])

    long_signal = (
        h1_trend_up(frame, i)
        and long_impulse
        and distance_atr <= params["pullback_atr"]
        and float(frame["low"].iat[i]) <= float(frame["ema20"].iat[i])
        and float(frame["close"].iat[i]) > float(frame["ema20"].iat[i])
        and float(frame["close"].iat[i]) > float(frame["prev_close"].iat[i])
    )
    short_signal = (
        h1_trend_down(frame, i)
        and short_impulse
        and distance_atr <= params["pullback_atr"]
        and float(frame["high"].iat[i]) >= float(frame["ema20"].iat[i])
        and float(frame["close"].iat[i]) < float(frame["ema20"].iat[i])
        and float(frame["close"].iat[i]) < float(frame["prev_close"].iat[i])
    )

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None
