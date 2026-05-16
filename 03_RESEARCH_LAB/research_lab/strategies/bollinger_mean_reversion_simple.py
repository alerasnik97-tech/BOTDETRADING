from __future__ import annotations

from research_lab.strategies.common import add_general_params, stratified_sample_combinations


NAME = "bollinger_mean_reversion_simple"
WARMUP_BARS = 120


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "bb_period": [20, 30],
            "bb_std": [2.0, 2.2, 2.5],
            "adx_max": [18, 20, 22],
            "stop_atr": [1.0, 1.2, 1.5],
            "tp_mode": ["midband", "rr"],
            "target_rr": [1.0, 1.2, 1.5],
        }
    )


def parameter_grid(max_combinations: int = 8, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    if float(frame["adx14"].iat[i]) > float(params["adx_max"]):
        return None

    suffix = f"{params['bb_period']}_{str(params['bb_std']).replace('.', '_')}"
    lower = float(frame[f"bb_lower_{suffix}"].iat[i])
    upper = float(frame[f"bb_upper_{suffix}"].iat[i])
    mid = float(frame[f"bb_mid_{suffix}"].iat[i])
    close = float(frame["close"].iat[i])

    if close < lower:
        signal = {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["tp_mode"] == "midband":
            signal["target_mode"] = "price"
            signal["target_price"] = mid
        else:
            signal["target_rr"] = params["target_rr"]
        return signal

    if close > upper:
        signal = {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["tp_mode"] == "midband":
            signal["target_mode"] = "price"
            signal["target_price"] = mid
        else:
            signal["target_rr"] = params["target_rr"]
        return signal
    return None
