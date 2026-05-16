from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, day_range_ok, h1_trend_down, h1_trend_up


NAME = "compression_breakout_m15"
WARMUP_BARS = 50


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"compression_bars": 4, "compression_atr_mult": 0.8, "use_h1_context": False, "stop_mode": "price", "stop_atr": 1.0, "target_rr": 1.5, "break_even_enabled": False},
        {"compression_bars": 4, "compression_atr_mult": 1.0, "use_h1_context": True, "stop_mode": "atr", "stop_atr": 1.0, "target_rr": 2.0, "break_even_enabled": False},
        {"compression_bars": 6, "compression_atr_mult": 0.8, "use_h1_context": False, "stop_mode": "price", "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": False},
        {"compression_bars": 6, "compression_atr_mult": 1.0, "use_h1_context": True, "stop_mode": "atr", "stop_atr": 1.2, "target_rr": 2.0, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    bars = params["compression_bars"]
    comp_high = float(frame[f"donchian_high_{bars}"].iat[i])
    comp_low = float(frame[f"donchian_low_{bars}"].iat[i])
    comp_range_atr = float(frame[f"donchian_range_atr_{bars}"].iat[i])
    if not (
        atr > 0
        and day_range_ok(frame, i)
        and 0.25 <= comp_range_atr <= params["compression_atr_mult"]
        and candle_is_not_noisy(frame, i, 1.5)
    ):
        return None

    long_signal = float(frame["close"].iat[i]) > comp_high and float(frame["prev_close"].iat[i]) <= comp_high
    short_signal = float(frame["close"].iat[i]) < comp_low and float(frame["prev_close"].iat[i]) >= comp_low

    if params["use_h1_context"]:
        long_signal = long_signal and h1_trend_up(frame, i)
        short_signal = short_signal and h1_trend_down(frame, i)

    if long_signal:
        signal = {"direction": "long", "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
        if params["stop_mode"] == "price":
            signal.update({"stop_mode": "price", "stop_price": comp_low})
        else:
            signal.update({"stop_mode": "atr", "stop_atr": params["stop_atr"]})
        return signal
    if short_signal:
        signal = {"direction": "short", "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
        if params["stop_mode"] == "price":
            signal.update({"stop_mode": "price", "stop_price": comp_high})
        else:
            signal.update({"stop_mode": "atr", "stop_atr": params["stop_atr"]})
        return signal
    return None
