from __future__ import annotations

from research_lab.strategies.common import add_general_params, candle_not_extended, day_range_ok, h1_filter_passes, stratified_sample_combinations


NAME = "bollinger_mean_reversion_adx_low"
WARMUP_BARS = 120


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "bb_period": [20, 30],
            "bb_std": [1.8, 2.0, 2.2, 2.5],
            "rsi_period": [7, 14],
            "rsi_oversold": [20, 25, 30],
            "rsi_overbought": [70, 75, 80],
            "adx_max": [18, 20, 22],
            "stop_atr": [1.0, 1.2, 1.5],
            "tp_mode": ["midband", "fixed_rr"],
            "fixed_rr": [1.0, 1.2, 1.5],
            "day_range_min_atr": [0.6],
            "day_range_max_atr": [2.2],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    if not (atr > 0 and candle_not_extended(frame, i, 1.5) and day_range_ok(frame, i, params["day_range_min_atr"], params["day_range_max_atr"])):
        return None

    suffix = f"{params['bb_period']}_{str(params['bb_std']).replace('.', '_')}"
    lower = float(frame[f"bb_lower_{suffix}"].iat[i])
    upper = float(frame[f"bb_upper_{suffix}"].iat[i])
    mid = float(frame[f"bb_mid_{suffix}"].iat[i])
    adx_value = float(frame["adx14"].iat[i])
    rsi_col = "rsi14" if params["rsi_period"] == 14 else "rsi7"
    rsi_value = float(frame[rsi_col].iat[i])
    if adx_value > params["adx_max"]:
        return None

    if params["use_h1_context"]:
        h1_ok_long = not h1_filter_passes(frame, i, True, "short")
        h1_ok_short = not h1_filter_passes(frame, i, True, "long")
    else:
        h1_ok_long = True
        h1_ok_short = True

    long_signal = (
        h1_ok_long
        and float(frame["low"].iat[i]) <= lower
        and float(frame["close"].iat[i]) >= lower
        and float(frame["close"].iat[i]) > float(frame["prev_close"].iat[i])
        and rsi_value <= params["rsi_oversold"]
    )
    short_signal = (
        h1_ok_short
        and float(frame["high"].iat[i]) >= upper
        and float(frame["close"].iat[i]) <= upper
        and float(frame["close"].iat[i]) < float(frame["prev_close"].iat[i])
        and rsi_value >= params["rsi_overbought"]
    )

    if long_signal:
        signal = {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["tp_mode"] == "midband":
            signal.update({"target_mode": "price", "target_price": mid})
        else:
            signal.update({"target_rr": params["fixed_rr"]})
        return signal
    if short_signal:
        signal = {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["tp_mode"] == "midband":
            signal.update({"target_mode": "price", "target_price": mid})
        else:
            signal.update({"target_rr": params["fixed_rr"]})
        return signal
    return None
