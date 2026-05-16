from __future__ import annotations

import math

from research_lab.strategies.common import stratified_sample_combinations


NAME = "rsi2_stochastic_range_reversion"
WARMUP_BARS = 80


def parameter_space() -> dict[str, list]:
    return {
        "rsi_period": [2, 3, 5],
        "rsi_oversold": [5, 10, 15],
        "rsi_overbought": [85, 90, 95],
        "stoch_k": [5, 7, 9],
        "stoch_d": [3],
        "stoch_oversold": [15, 20],
        "stoch_overbought": [80, 85],
        "adx_max": [16, 18, 20],
        "mean_filter": ["ema20", "ema30", "bb_mid"],
        "atr_stop": [1.0, 1.2, 1.5],
        "target_mode": ["mean_reversion", "rr"],
        "target_rr": [1.0, 1.2, 1.5],
        "break_even_at_r": [None, 1.0],
        "session_name": ["light_fixed"],
    }


def parameter_grid(max_combinations: int = 12, seed: int = 42) -> list[dict]:
    return stratified_sample_combinations(parameter_space(), max_combinations, seed)


def default_params() -> dict:
    return parameter_grid(1)[0]


def _mean_value(frame, i: int, mean_filter: str) -> float:
    if mean_filter == "ema20":
        return float(frame["ema20"].iat[i])
    if mean_filter == "ema30":
        return float(frame["ema30"].iat[i])
    return float(frame["bb_mid_20_2_0"].iat[i])


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    adx_value = float(frame["adx14"].iat[i])
    if not (math.isfinite(atr) and atr > 0 and math.isfinite(adx_value) and adx_value <= float(params["adx_max"])):
        return None

    mean_value = _mean_value(frame, i, str(params["mean_filter"]))
    close = float(frame["close"].iat[i])
    if not math.isfinite(mean_value):
        return None

    rsi_value = float(frame[f"rsi{params['rsi_period']}"].iat[i])
    stoch_k_col = f"stoch_k_{params['stoch_k']}"
    stoch_d_col = f"stoch_d_{params['stoch_k']}_{params['stoch_d']}"
    stoch_k_now = float(frame[stoch_k_col].iat[i])
    stoch_d_now = float(frame[stoch_d_col].iat[i])
    stoch_k_prev = float(frame[stoch_k_col].iat[i - 1])
    stoch_d_prev = float(frame[stoch_d_col].iat[i - 1])
    if not all(math.isfinite(value) for value in (rsi_value, stoch_k_now, stoch_d_now, stoch_k_prev, stoch_d_prev)):
        return None

    mean_gap_atr = (mean_value - close) / atr
    long_signal = (
        mean_gap_atr >= 0.15
        and rsi_value <= float(params["rsi_oversold"])
        and min(stoch_k_prev, stoch_d_prev) <= float(params["stoch_oversold"])
        and stoch_k_now > stoch_d_now
        and stoch_k_now > stoch_k_prev
    )
    short_signal = (
        mean_gap_atr <= -0.15
        and rsi_value >= float(params["rsi_overbought"])
        and max(stoch_k_prev, stoch_d_prev) >= float(params["stoch_overbought"])
        and stoch_k_now < stoch_d_now
        and stoch_k_now < stoch_k_prev
    )

    if long_signal:
        signal_payload = {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["atr_stop"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["target_mode"] == "mean_reversion":
            signal_payload["target_mode"] = "price"
            signal_payload["target_price"] = mean_value
        else:
            signal_payload["target_rr"] = params["target_rr"]
        return signal_payload

    if short_signal:
        signal_payload = {
            "direction": "short",
            "stop_mode": "atr",
            "stop_atr": params["atr_stop"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["target_mode"] == "mean_reversion":
            signal_payload["target_mode"] = "price"
            signal_payload["target_price"] = mean_value
        else:
            signal_payload["target_rr"] = params["target_rr"]
        return signal_payload

    return None
