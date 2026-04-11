from __future__ import annotations

NAME = "bollinger_squeeze_breakout"
WARMUP_BARS = 30


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"bb_std": 2.0, "squeeze_width_atr_max": 1.2, "breakout_candle_atr_max": 1.2, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"bb_std": 2.0, "squeeze_width_atr_max": 1.5, "breakout_candle_atr_max": 1.5, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": False},
        {"bb_std": 2.2, "squeeze_width_atr_max": 1.2, "breakout_candle_atr_max": 1.2, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": True},
        {"bb_std": 2.2, "squeeze_width_atr_max": 1.5, "breakout_candle_atr_max": 1.5, "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    suffix = "2_2" if abs(params["bb_std"] - 2.2) < 1e-9 else "2_0"
    upper = frame[f"bb_upper_20_{suffix}"].iat[i]
    lower = frame[f"bb_lower_20_{suffix}"].iat[i]
    width_atr = frame[f"bb_width_atr_20_{suffix}"].iat[i]
    if width_atr > params["squeeze_width_atr_max"] or frame["range_atr"].iat[i] > params["breakout_candle_atr_max"]:
        return None

    long_signal = frame["close"].iat[i] > upper and frame["prev_close"].iat[i] <= upper
    short_signal = frame["close"].iat[i] < lower and frame["prev_close"].iat[i] >= lower
    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

