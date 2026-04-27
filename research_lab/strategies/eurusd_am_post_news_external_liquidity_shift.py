from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import finite, is_in_session


NAME = "eurusd_am_post_news_external_liquidity_shift"
WARMUP_BARS = 40
EXPLICIT_TIMEFRAME = "M3"


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "post_news_external_liquidity_shift_v1",
        "session_name": "ny_open",
        "target_rr": 2.1,
        "break_even_at_r": 1.2,
        "max_hold_bars": 21,
        "cooldown_bars": 0,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _trade_window_open(frame: pd.DataFrame, i: int) -> bool:
    return bool(frame["els_trade_window_open"].iat[i]) if "els_trade_window_open" in frame.columns else False


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 1 or not is_in_session(ts, str(params["session_name"])) or not _trade_window_open(frame, i):
        return None
    if "els_signal" not in frame.columns or not bool(frame["els_signal"].iat[i]):
        return None

    direction = str(frame["els_direction"].iat[i]).strip().lower()
    stop_price = float(frame["els_stop_price"].iat[i])
    if direction not in {"long", "short"} or not finite(stop_price):
        return None

    return {
        "direction": direction,
        "entry_mode": "market",
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "rr",
        "target_rr": float(frame["els_target_rr"].iat[i]) if "els_target_rr" in frame.columns else float(params["target_rr"]),
        "max_hold_bars": int(frame["els_max_hold_bars"].iat[i]) if "els_max_hold_bars" in frame.columns else int(params["max_hold_bars"]),
        "break_even_at_r": float(frame["els_break_even_at_r"].iat[i]) if "els_break_even_at_r" in frame.columns else float(params["break_even_at_r"]),
        "cooldown_bars": int(params.get("cooldown_bars", 0)),
        "session_name": str(params["session_name"]),
    }
