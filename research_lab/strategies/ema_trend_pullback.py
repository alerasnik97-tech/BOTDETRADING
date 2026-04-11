from __future__ import annotations

from research_lab.strategies.common import add_general_params, candle_not_extended, stratified_sample_combinations


NAME = "ema_trend_pullback"
WARMUP_BARS = 250


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "ema_fast": [20, 30, 50],
            "ema_slow": [100, 150, 200],
            "ema_pullback": [10, 20],
            "adx_min": [18, 20, 22],
            "stop_atr": [1.2, 1.5, 2.0],
            "target_rr": [1.2, 1.5, 2.0],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (atr > 0 and candle_not_extended(frame, i, 1.5)):
        return None

    ema_fast = float(frame[f"ema{params['ema_fast']}"].iat[i])
    ema_slow = float(frame[f"ema{params['ema_slow']}"].iat[i])
    ema_pullback = float(frame[f"ema{params['ema_pullback']}"].iat[i])
    adx_value = float(frame["adx14"].iat[i])

    long_bias = ema_fast > ema_slow and adx_value >= params["adx_min"]
    short_bias = ema_fast < ema_slow and adx_value >= params["adx_min"]

    long_signal = (
        long_bias
        and float(frame["low"].iat[i]) <= ema_pullback
        and float(frame["close"].iat[i]) > ema_pullback
        and float(frame["close"].iat[i]) > float(frame["prev_high"].iat[i])
    )
    short_signal = (
        short_bias
        and float(frame["high"].iat[i]) >= ema_pullback
        and float(frame["close"].iat[i]) < ema_pullback
        and float(frame["close"].iat[i]) < float(frame["prev_low"].iat[i])
    )

    if long_signal:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
    if short_signal:
        return {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
    return None
