from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import finite, is_in_session


NAME = "eurusd_ltf_objective_entry_replacement_ecb"
WARMUP_BARS = 0
EXPLICIT_TIMEFRAME = "M3"


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "ecb_fixed_stage2_v1",
        "session_name": "all_day",
        "target_rr": 1.5,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if not is_in_session(ts, str(params["session_name"])):
        return None
    if "ecb_signal" not in frame.columns or not bool(frame["ecb_signal"].iat[i]):
        return None

    direction = str(frame["ecb_direction"].iat[i]).strip().lower()
    stop_price = float(frame["ecb_stop_price"].iat[i])
    stop_entry_price = float(frame["ecb_stop_entry_price"].iat[i])
    if direction not in {"long", "short"}:
        return None
    if not finite(stop_price) or not finite(stop_entry_price):
        return None

    return {
        "direction": direction,
        "entry_mode": "stop",
        "stop_entry_price": stop_entry_price,
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "rr",
        "target_rr": float(frame["ecb_target_rr"].iat[i]) if "ecb_target_rr" in frame.columns else float(params["target_rr"]),
        "session_name": str(params["session_name"]),
    }
