from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, close_reclaims_down, close_reclaims_up, day_range_ok, finite, h1_trend_down, h1_trend_up


NAME = "liquidity_sweep_continuation"
WARMUP_BARS = 80


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"structure_bars": 4, "impulse_atr": 0.8, "sweep_buffer_atr": 0.10, "stop_buffer_atr": 0.20, "target_rr": 1.5, "break_even_enabled": False},
        {"structure_bars": 4, "impulse_atr": 1.0, "sweep_buffer_atr": 0.15, "stop_buffer_atr": 0.25, "target_rr": 1.8, "break_even_enabled": False},
        {"structure_bars": 6, "impulse_atr": 0.8, "sweep_buffer_atr": 0.10, "stop_buffer_atr": 0.20, "target_rr": 1.5, "break_even_enabled": False},
        {"structure_bars": 6, "impulse_atr": 1.0, "sweep_buffer_atr": 0.15, "stop_buffer_atr": 0.25, "target_rr": 1.8, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (atr > 0 and candle_is_not_noisy(frame, i, 1.5) and day_range_ok(frame, i)):
        return None

    bars = params["structure_bars"]
    structure_high = float(frame[f"m15_range_high_{bars}"].iat[i])
    structure_low = float(frame[f"m15_range_low_{bars}"].iat[i])
    prev_day_high = float(frame["prev_day_high"].iat[i])
    prev_day_low = float(frame["prev_day_low"].iat[i])
    sweep_buffer = atr * params["sweep_buffer_atr"]
    stop_buffer = atr * params["stop_buffer_atr"]
    long_impulse = float(frame["m15_close_change_atr_3"].iat[i]) >= params["impulse_atr"] and float(frame["m15_close"].iat[i]) >= float(frame["m15_ema20"].iat[i])
    short_impulse = float(frame["m15_close_change_atr_3"].iat[i]) <= -params["impulse_atr"] and float(frame["m15_close"].iat[i]) <= float(frame["m15_ema20"].iat[i])

    long_level = None
    if finite(prev_day_low) and float(frame["low"].iat[i]) <= prev_day_low - sweep_buffer and close_reclaims_up(frame, i, prev_day_low):
        long_level = prev_day_low
    elif finite(structure_low) and float(frame["low"].iat[i]) <= structure_low - sweep_buffer and close_reclaims_up(frame, i, structure_low):
        long_level = structure_low

    short_level = None
    if finite(prev_day_high) and float(frame["high"].iat[i]) >= prev_day_high + sweep_buffer and close_reclaims_down(frame, i, prev_day_high):
        short_level = prev_day_high
    elif finite(structure_high) and float(frame["high"].iat[i]) >= structure_high + sweep_buffer and close_reclaims_down(frame, i, structure_high):
        short_level = structure_high

    if h1_trend_up(frame, i) and long_impulse and long_level is not None:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": min(float(frame["low"].iat[i]), float(long_level)) - stop_buffer,
            "target_rr": params["target_rr"],
            "break_even_enabled": params["break_even_enabled"],
        }
    if h1_trend_down(frame, i) and short_impulse and short_level is not None:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": max(float(frame["high"].iat[i]), float(short_level)) + stop_buffer,
            "target_rr": params["target_rr"],
            "break_even_enabled": params["break_even_enabled"],
        }
    return None
