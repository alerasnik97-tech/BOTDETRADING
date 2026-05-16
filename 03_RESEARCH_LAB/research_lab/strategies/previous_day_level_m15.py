from __future__ import annotations

from research_lab.strategies.common import candle_not_extended, day_range_ok, finite, h1_filter_passes


NAME = "previous_day_level_m15"
WARMUP_BARS = 50


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"entry_mode": "rejection", "use_h1_context": False, "stop_buffer_atr": 0.15, "target_rr": 1.5, "break_even_enabled": False},
        {"entry_mode": "rejection", "use_h1_context": True, "stop_buffer_atr": 0.25, "target_rr": 2.0, "break_even_enabled": False},
        {"entry_mode": "breakout", "use_h1_context": False, "stop_buffer_atr": 0.15, "target_rr": 1.5, "break_even_enabled": False},
        {"entry_mode": "breakout", "use_h1_context": True, "stop_buffer_atr": 0.25, "target_rr": 2.0, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    prev_day_high = float(frame["prev_day_high"].iat[i])
    prev_day_low = float(frame["prev_day_low"].iat[i])
    if not (
        atr > 0
        and finite(prev_day_high)
        and finite(prev_day_low)
        and day_range_ok(frame, i, 0.6, 3.0)
        and candle_not_extended(frame, i, 1.5)
    ):
        return None

    close_price = float(frame["close"].iat[i])
    prev_close = float(frame["prev_close"].iat[i])
    high_price = float(frame["high"].iat[i])
    low_price = float(frame["low"].iat[i])
    stop_buffer = atr * params["stop_buffer_atr"]

    if params["entry_mode"] == "rejection":
        long_signal = low_price <= prev_day_low and close_price > prev_day_low and close_price > prev_close
        short_signal = high_price >= prev_day_high and close_price < prev_day_high and close_price < prev_close
    else:
        long_signal = close_price > prev_day_high and prev_close <= prev_day_high and abs(close_price - prev_day_high) <= 0.6 * atr
        short_signal = close_price < prev_day_low and prev_close >= prev_day_low and abs(close_price - prev_day_low) <= 0.6 * atr

    if params["use_h1_context"]:
        if params["entry_mode"] == "breakout":
            long_signal = long_signal and h1_filter_passes(frame, i, True, "long")
            short_signal = short_signal and h1_filter_passes(frame, i, True, "short")
        else:
            long_signal = long_signal and not h1_filter_passes(frame, i, True, "short")
            short_signal = short_signal and not h1_filter_passes(frame, i, True, "long")

    if long_signal:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": min(low_price, prev_day_low) - stop_buffer,
            "target_rr": params["target_rr"],
            "break_even_enabled": params["break_even_enabled"],
        }
    if short_signal:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": max(high_price, prev_day_high) + stop_buffer,
            "target_rr": params["target_rr"],
            "break_even_enabled": params["break_even_enabled"],
        }
    return None
