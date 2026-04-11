from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, day_range_ok


NAME = "session_momentum_shift"
WARMUP_BARS = 80


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"structure_bars": 4, "shift_atr": 0.3, "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"structure_bars": 4, "shift_atr": 0.5, "stop_atr": 1.0, "target_rr": 1.8, "break_even_enabled": False},
        {"structure_bars": 6, "shift_atr": 0.3, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": False},
        {"structure_bars": 6, "shift_atr": 0.5, "stop_atr": 1.2, "target_rr": 1.8, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (atr > 0 and candle_is_not_noisy(frame, i, 1.5) and day_range_ok(frame, i)):
        return None

    bars = params["structure_bars"]
    break_high = float(frame[f"m15_range_high_{bars}"].iat[i])
    break_low = float(frame[f"m15_range_low_{bars}"].iat[i])
    shift_up = float(frame["m15_close"].iat[i]) > float(frame["m15_ema20"].iat[i]) and float(frame["m15_prev_close"].iat[i]) <= float(frame["m15_ema20"].iat[i]) and float(frame["m15_close_change_atr_2"].iat[i]) >= params["shift_atr"]
    shift_down = float(frame["m15_close"].iat[i]) < float(frame["m15_ema20"].iat[i]) and float(frame["m15_prev_close"].iat[i]) >= float(frame["m15_ema20"].iat[i]) and float(frame["m15_close_change_atr_2"].iat[i]) <= -params["shift_atr"]

    long_signal = shift_up and float(frame["close"].iat[i]) > break_high and float(frame["prev_close"].iat[i]) <= break_high and float(frame["close"].iat[i]) >= float(frame["ema20"].iat[i])
    short_signal = shift_down and float(frame["close"].iat[i]) < break_low and float(frame["prev_close"].iat[i]) >= break_low and float(frame["close"].iat[i]) <= float(frame["ema20"].iat[i])

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None
