from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, day_range_ok, h1_trend_down, h1_trend_up


NAME = "session_range_breakout_m15"
WARMUP_BARS = 40


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"entry_mode": "direct", "use_h1_context": False, "stop_mode": "structure", "stop_atr": 1.0, "target_rr": 1.5, "breakout_candle_atr_max": 1.2, "break_even_enabled": False},
        {"entry_mode": "direct", "use_h1_context": True, "stop_mode": "atr", "stop_atr": 1.0, "target_rr": 2.0, "breakout_candle_atr_max": 1.5, "break_even_enabled": False},
        {"entry_mode": "retest", "use_h1_context": False, "stop_mode": "structure", "stop_atr": 1.2, "target_rr": 1.5, "breakout_candle_atr_max": 1.2, "break_even_enabled": False},
        {"entry_mode": "retest", "use_h1_context": True, "stop_mode": "atr", "stop_atr": 1.2, "target_rr": 2.0, "breakout_candle_atr_max": 1.5, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    ts = frame.index[i]
    if (ts.hour * 60 + ts.minute) < 13 * 60:
        return None
    atr = float(frame["atr14"].iat[i])
    range_high = float(frame["session_range_high_13_00"].iat[i])
    range_low = float(frame["session_range_low_13_00"].iat[i])
    complete = bool(frame["session_range_complete_13_00"].iat[i])
    if not (complete and atr > 0 and day_range_ok(frame, i) and candle_is_not_noisy(frame, i, params["breakout_candle_atr_max"])):
        return None

    if params["entry_mode"] == "direct":
        long_signal = float(frame["close"].iat[i]) > range_high and float(frame["prev_close"].iat[i]) <= range_high
        short_signal = float(frame["close"].iat[i]) < range_low and float(frame["prev_close"].iat[i]) >= range_low
    else:
        long_signal = (
            float(frame["prev_close"].iat[i]) > range_high
            and float(frame["low"].iat[i]) <= range_high
            and float(frame["close"].iat[i]) > range_high
        )
        short_signal = (
            float(frame["prev_close"].iat[i]) < range_low
            and float(frame["high"].iat[i]) >= range_low
            and float(frame["close"].iat[i]) < range_low
        )

    if params["use_h1_context"]:
        long_signal = long_signal and h1_trend_up(frame, i)
        short_signal = short_signal and h1_trend_down(frame, i)

    if long_signal:
        signal = {"direction": "long", "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
        if params["stop_mode"] == "structure":
            signal.update({"stop_mode": "price", "stop_price": range_low})
        else:
            signal.update({"stop_mode": "atr", "stop_atr": params["stop_atr"]})
        return signal
    if short_signal:
        signal = {"direction": "short", "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
        if params["stop_mode"] == "structure":
            signal.update({"stop_mode": "price", "stop_price": range_high})
        else:
            signal.update({"stop_mode": "atr", "stop_atr": params["stop_atr"]})
        return signal
    return None
