from __future__ import annotations

from research_lab.strategies.common import stratified_sample_combinations, add_general_params


NAME = "keltner_volatility_expansion_simple"
WARMUP_BARS = 120


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "ema_base": [20, 30],
            "atr_mult_keltner": [1.5, 2.0],
            "ema_filter": [100, 200],
            "expansion_atr_min": [1.0, 1.2, 1.5],
            "stop_atr": [1.0, 1.5],
            "target_rr": [1.2, 1.5, 2.0],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    suffix = f"{params['ema_base']}_{str(params['atr_mult_keltner']).replace('.', '_')}"
    kc_upper = float(frame[f"kc_upper_{suffix}"].iat[i])
    kc_lower = float(frame[f"kc_lower_{suffix}"].iat[i])
    prev_close_1 = float(frame["close"].iat[i - 1])
    prev_close_2 = float(frame["close"].iat[i - 2])
    prev_upper_1 = float(frame[f"kc_upper_{suffix}"].iat[i - 1])
    prev_lower_1 = float(frame[f"kc_lower_{suffix}"].iat[i - 1])
    prev_upper_2 = float(frame[f"kc_upper_{suffix}"].iat[i - 2])
    prev_lower_2 = float(frame[f"kc_lower_{suffix}"].iat[i - 2])
    expansion_ok = float(frame["range_atr"].iat[i]) >= float(params["expansion_atr_min"])
    close = float(frame["close"].iat[i])
    ema_filter = float(frame[f"ema{params['ema_filter']}"].iat[i])

    compressed = (
        prev_lower_1 <= prev_close_1 <= prev_upper_1
        and prev_lower_2 <= prev_close_2 <= prev_upper_2
    )

    if compressed and expansion_ok and close > kc_upper and close > ema_filter:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
    if compressed and expansion_ok and close < kc_lower and close < ema_filter:
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
