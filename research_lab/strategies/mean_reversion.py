from __future__ import annotations

NAME = "mean_reversion"
WARMUP_BARS = 20


def parameter_grid(max_combinations: int = 8) -> list[dict]:
    combos = [
        {"rsi_period": 9, "bb_std": 2.0, "rsi_low": 30, "rsi_high": 70, "signal_candle_atr_max": 1.2, "stop_atr": 1.0, "target_rr": 1.2, "break_even_enabled": False},
        {"rsi_period": 9, "bb_std": 2.2, "rsi_low": 35, "rsi_high": 65, "signal_candle_atr_max": 1.5, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"rsi_period": 14, "bb_std": 2.0, "rsi_low": 30, "rsi_high": 70, "signal_candle_atr_max": 1.2, "stop_atr": 1.2, "target_rr": 1.2, "break_even_enabled": True},
        {"rsi_period": 14, "bb_std": 2.2, "rsi_low": 35, "rsi_high": 65, "signal_candle_atr_max": 1.5, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": True},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    suffix = "2_2" if abs(params["bb_std"] - 2.2) < 1e-9 else "2_0"
    upper = frame[f"bb_upper_20_{suffix}"].iat[i]
    lower = frame[f"bb_lower_20_{suffix}"].iat[i]
    width_atr = frame[f"bb_width_atr_20_{suffix}"].iat[i]
    rsi_col = "rsi14" if params["rsi_period"] == 14 else "rsi9"
    rsi_value = frame[rsi_col].iat[i]

    if not (1.0 <= width_atr <= 4.5 and frame["range_atr"].iat[i] <= params["signal_candle_atr_max"]):
        return None

    long_signal = frame["low"].iat[i] <= lower and frame["close"].iat[i] >= lower and rsi_value <= params["rsi_low"] and frame["close"].iat[i] > frame["prev_close"].iat[i]
    short_signal = frame["high"].iat[i] >= upper and frame["close"].iat[i] <= upper and rsi_value >= params["rsi_high"] and frame["close"].iat[i] < frame["prev_close"].iat[i]

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None

