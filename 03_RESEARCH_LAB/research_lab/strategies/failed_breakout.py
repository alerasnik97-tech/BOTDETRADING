from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, close_reclaims_down, close_reclaims_up, day_range_ok, finite, h1_trend_down, h1_trend_up


NAME = "failed_breakout"
WARMUP_BARS = 80


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"structure_bars": 4, "sweep_buffer_atr": 0.10, "stop_buffer_atr": 0.15, "target_rr": 1.5, "break_even_enabled": False},
        {"structure_bars": 4, "sweep_buffer_atr": 0.20, "stop_buffer_atr": 0.20, "target_rr": 1.8, "break_even_enabled": False},
        {"structure_bars": 6, "sweep_buffer_atr": 0.10, "stop_buffer_atr": 0.15, "target_rr": 1.5, "break_even_enabled": False},
        {"structure_bars": 6, "sweep_buffer_atr": 0.20, "stop_buffer_atr": 0.20, "target_rr": 1.8, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (atr > 0 and candle_is_not_noisy(frame, i, 1.5) and day_range_ok(frame, i)):
        return None

    if h1_trend_up(frame, i) or h1_trend_down(frame, i):
        return None

    bars = params["structure_bars"]
    structure_high = float(frame[f"m15_range_high_{bars}"].iat[i])
    structure_low = float(frame[f"m15_range_low_{bars}"].iat[i])
    prev_day_high = float(frame["prev_day_high"].iat[i])
    prev_day_low = float(frame["prev_day_low"].iat[i])
    sweep_buffer = atr * params["sweep_buffer_atr"]
    stop_buffer = atr * params["stop_buffer_atr"]

    long_signal = (
        (finite(prev_day_low) and float(frame["low"].iat[i]) <= prev_day_low - sweep_buffer and close_reclaims_up(frame, i, prev_day_low))
        or (finite(structure_low) and float(frame["low"].iat[i]) <= structure_low - sweep_buffer and close_reclaims_up(frame, i, structure_low))
    )
    short_signal = (
        (finite(prev_day_high) and float(frame["high"].iat[i]) >= prev_day_high + sweep_buffer and close_reclaims_down(frame, i, prev_day_high))
        or (finite(structure_high) and float(frame["high"].iat[i]) >= structure_high + sweep_buffer and close_reclaims_down(frame, i, structure_high))
    )

    if long_signal:
        return {
            "direction": "long",
            "stop_mode": "price",
            "stop_price": float(frame["low"].iat[i]) - stop_buffer,
            "target_rr": params["target_rr"],
            "break_even_enabled": params["break_even_enabled"],
        }
    if short_signal:
        return {
            "direction": "short",
            "stop_mode": "price",
            "stop_price": float(frame["high"].iat[i]) + stop_buffer,
            "target_rr": params["target_rr"],
            "break_even_enabled": params["break_even_enabled"],
        }
    return None
