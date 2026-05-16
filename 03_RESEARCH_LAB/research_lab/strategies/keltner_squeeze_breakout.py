from __future__ import annotations

from research_lab.strategies.common import add_general_params, candle_not_extended, day_range_ok, h1_filter_passes, stratified_sample_combinations


NAME = "keltner_squeeze_breakout"
WARMUP_BARS = 120


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "kc_ema_length": [20, 30],
            "kc_atr_mult": [1.5, 2.0],
            "bb_squeeze_std": [1.8, 2.0],
            "ema_filter": [50, 100, 200],
            "expansion_atr_min": [1.0, 1.2, 1.5],
            "stop_atr": [1.0, 1.5, 2.0],
            "target_rr": [1.2, 1.5, 2.0, 2.5],
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
        and candle_not_extended(frame, i, max(1.8, params["expansion_atr_min"] + 0.3))
        and day_range_ok(frame, i, params["day_range_min_atr"], params["day_range_max_atr"])
    ):
        return None

    bb_suffix = f"20_{str(params['bb_squeeze_std']).replace('.', '_')}"
    kc_suffix = f"{params['kc_ema_length']}_{str(params['kc_atr_mult']).replace('.', '_')}"
    bb_upper = float(frame[f"bb_upper_{bb_suffix}"].iat[i - 1])
    bb_lower = float(frame[f"bb_lower_{bb_suffix}"].iat[i - 1])
    kc_upper = float(frame[f"kc_upper_{kc_suffix}"].iat[i - 1])
    kc_lower = float(frame[f"kc_lower_{kc_suffix}"].iat[i - 1])
    squeeze = bb_upper <= kc_upper and bb_lower >= kc_lower
    expansion = float(frame["range_atr"].iat[i]) >= params["expansion_atr_min"]
    ema_filter = float(frame[f"ema{params['ema_filter']}"].iat[i])

    long_signal = (
        squeeze
        and expansion
        and float(frame["close"].iat[i]) > float(frame[f"bb_upper_{bb_suffix}"].iat[i])
        and float(frame["close"].iat[i]) > ema_filter
        and h1_filter_passes(frame, i, params["use_h1_context"], "long", params["ema_filter"])
    )
    short_signal = (
        squeeze
        and expansion
        and float(frame["close"].iat[i]) < float(frame[f"bb_lower_{bb_suffix}"].iat[i])
        and float(frame["close"].iat[i]) < ema_filter
        and h1_filter_passes(frame, i, params["use_h1_context"], "short", params["ema_filter"])
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
