from __future__ import annotations

from research_lab.strategies.common import add_general_params, candle_not_extended, stratified_sample_combinations


NAME = "supertrend_ema_filter"
WARMUP_BARS = 250


def parameter_space() -> dict[str, list]:
    return add_general_params(
        {
            "atr_period": [7, 10, 14],
            "supertrend_mult": [2.0, 2.5, 3.0],
            "ema_filter": [100, 200],
            "entry_mode": ["flip", "continuation"],
            "stop_mode": ["atr", "supertrend"],
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

    suffix = f"{params['atr_period']}_{str(params['supertrend_mult']).replace('.', '_')}"
    st_dir = float(frame[f"supertrend_dir_{suffix}"].iat[i])
    st_dir_prev = float(frame[f"supertrend_dir_{suffix}"].iat[i - 1])
    st_line = float(frame[f"supertrend_line_{suffix}"].iat[i])
    ema_filter = float(frame[f"ema{params['ema_filter']}"].iat[i])

    if params["entry_mode"] == "flip":
        long_signal = st_dir == 1.0 and st_dir_prev != 1.0 and float(frame["close"].iat[i]) > ema_filter
        short_signal = st_dir == -1.0 and st_dir_prev != -1.0 and float(frame["close"].iat[i]) < ema_filter
    else:
        long_signal = (
            st_dir == 1.0
            and st_dir_prev == 1.0
            and float(frame["low"].iat[i]) <= st_line
            and float(frame["close"].iat[i]) > st_line
            and float(frame["close"].iat[i]) > ema_filter
        )
        short_signal = (
            st_dir == -1.0
            and st_dir_prev == -1.0
            and float(frame["high"].iat[i]) >= st_line
            and float(frame["close"].iat[i]) < st_line
            and float(frame["close"].iat[i]) < ema_filter
        )

    if long_signal:
        signal = {
            "direction": "long",
            "stop_mode": "atr" if params["stop_mode"] == "atr" else "price",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["stop_mode"] == "atr":
            signal["stop_atr"] = params["stop_atr"]
        else:
            signal["stop_price"] = st_line
        return signal
    if short_signal:
        signal = {
            "direction": "short",
            "stop_mode": "atr" if params["stop_mode"] == "atr" else "price",
            "target_rr": params["target_rr"],
            "break_even_at_r": params["break_even_at_r"],
            "trailing_atr": False,
            "session_name": params["session_name"],
        }
        if params["stop_mode"] == "atr":
            signal["stop_atr"] = params["stop_atr"]
        else:
            signal["stop_price"] = st_line
        return signal
    return None
