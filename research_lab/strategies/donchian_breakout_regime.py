from __future__ import annotations

from research_lab.strategies.common import add_general_params, candle_not_extended, day_range_ok, h1_filter_passes, stratified_sample_combinations


NAME = "donchian_breakout_regime"
WARMUP_BARS = 120


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "donchian_bars": [20, 30, 40, 55],
            "ema_filter": [100, 200],
            "adx_min": [20, 25, 30],
            "breakout_candle_atr_max": [1.0, 1.2, 1.5],
            "stop_atr": [1.0, 1.5, 2.0],
            "target_rr": [1.2, 1.5, 2.0, 2.5],
            "trailing_atr": [False, True],
            "day_range_min_atr": [0.8],
            "day_range_max_atr": [3.5],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (
        atr > 0
        and candle_not_extended(frame, i, params["breakout_candle_atr_max"])
        and day_range_ok(frame, i, params["day_range_min_atr"], params["day_range_max_atr"])
    ):
        return None

    high_level = float(frame[f"donchian_high_{params['donchian_bars']}"].iat[i])
    low_level = float(frame[f"donchian_low_{params['donchian_bars']}"].iat[i])
    ema_filter = float(frame[f"ema{params['ema_filter']}"].iat[i])
    adx_value = float(frame["adx14"].iat[i])

    long_signal = (
        adx_value >= params["adx_min"]
        and float(frame["close"].iat[i]) > high_level
        and float(frame["prev_close"].iat[i]) <= high_level
        and float(frame["close"].iat[i]) > ema_filter
        and h1_filter_passes(frame, i, params["use_h1_context"], "long", params["ema_filter"], params["adx_min"])
    )
    short_signal = (
        adx_value >= params["adx_min"]
        and float(frame["close"].iat[i]) < low_level
        and float(frame["prev_close"].iat[i]) >= low_level
        and float(frame["close"].iat[i]) < ema_filter
        and h1_filter_passes(frame, i, params["use_h1_context"], "short", params["ema_filter"], params["adx_min"])
    )

    if long_signal:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": params["trailing_atr"],
            "session_name": params["session_name"],
        }
    if short_signal:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": params["trailing_atr"],
            "session_name": params["session_name"],
        }
    return None
