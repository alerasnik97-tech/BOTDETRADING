from __future__ import annotations

from research_lab.strategies.common import candle_is_not_noisy, finite


NAME = "bollinger_mean_reversion_m15"
WARMUP_BARS = 50


def parameter_grid(max_combinations: int = 4) -> list[dict]:
    combos = [
        {"use_h1_filter": False, "rsi_period": 9, "bb_std": 2.0, "stop_atr": 1.0, "target_rr": 1.2, "break_even_enabled": False},
        {"use_h1_filter": False, "rsi_period": 14, "bb_std": 2.2, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": False},
        {"use_h1_filter": True, "rsi_period": 9, "bb_std": 2.0, "stop_atr": 1.0, "target_rr": 1.2, "break_even_enabled": False},
        {"use_h1_filter": True, "rsi_period": 14, "bb_std": 2.2, "stop_atr": 1.2, "target_rr": 1.5, "break_even_enabled": False},
    ]
    return combos[:max_combinations]


def default_params() -> dict:
    return parameter_grid(1)[0]


def signal(frame, i: int, params: dict) -> dict | None:
    atr = float(frame["atr14"].iat[i])
    suffix = "2_2" if abs(params["bb_std"] - 2.2) < 1e-9 else "2_0"
    lower = float(frame[f"bb_lower_20_{suffix}"].iat[i])
    upper = float(frame[f"bb_upper_20_{suffix}"].iat[i])
    width_atr = float(frame[f"bb_width_atr_20_{suffix}"].iat[i])
    day_range_atr = float(frame["day_range_h1_atr"].iat[i])
    rsi_col = "rsi14" if params["rsi_period"] == 14 else "rsi9"
    rsi_value = float(frame[rsi_col].iat[i])
    if not (
        atr > 0
        and candle_is_not_noisy(frame, i, 1.5)
        and finite(width_atr)
        and 0.5 <= width_atr <= 3.0
        and finite(day_range_atr)
        and 0.35 <= day_range_atr <= 2.0
    ):
        return None

    if params["use_h1_filter"]:
        ema_gap_atr = abs(float(frame["h1_ema50"].iat[i]) - float(frame["h1_ema200"].iat[i])) / max(float(frame["h1_atr14"].iat[i]), 1e-9)
        if not (float(frame["h1_adx14"].iat[i]) <= 18 and ema_gap_atr <= 1.0):
            return None

    rsi_low = 35 if params["rsi_period"] == 9 else 30
    rsi_high = 65 if params["rsi_period"] == 9 else 70
    long_signal = float(frame["low"].iat[i]) <= lower and float(frame["close"].iat[i]) >= lower and float(frame["close"].iat[i]) > float(frame["prev_close"].iat[i]) and rsi_value <= rsi_low
    short_signal = float(frame["high"].iat[i]) >= upper and float(frame["close"].iat[i]) <= upper and float(frame["close"].iat[i]) < float(frame["prev_close"].iat[i]) and rsi_value >= rsi_high

    if long_signal:
        return {"direction": "long", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    if short_signal:
        return {"direction": "short", "stop_mode": "atr", "stop_atr": params["stop_atr"], "target_rr": params["target_rr"], "break_even_enabled": params["break_even_enabled"]}
    return None
