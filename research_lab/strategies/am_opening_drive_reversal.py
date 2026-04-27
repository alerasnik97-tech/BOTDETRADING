from __future__ import annotations

from typing import Any

import pandas as pd

from research_lab.strategies.common import finite, is_in_session


NAME = "am_opening_drive_reversal"
WARMUP_BARS = 40
EXPLICIT_TIMEFRAME = "M1"
PIP_SIZE = 0.0001


DEFAULT_GRID: list[dict[str, Any]] = [
    {
        "variant_label": "ny_open_drive_failure_reclaim",
        "session_name": "ny_open",
        "entry_start": "09:45",
        "entry_end": "10:30",
        "target_rr": 1.5,
        "cooldown_bars": 0,
        "retest_window_bars": 15,
        "stop_buffer_pips": 1.0,
    }
]


def parameter_space() -> dict[str, list[Any]]:
    return {}


def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict[str, Any]]:
    del seed
    return DEFAULT_GRID[:max_combinations]


def default_params() -> dict[str, Any]:
    return dict(DEFAULT_GRID[0])


def _minute_value(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _is_trade_window(ts: pd.Timestamp, params: dict[str, Any]) -> bool:
    start_hour, start_minute = (int(part) for part in str(params["entry_start"]).split(":"))
    end_hour, end_minute = (int(part) for part in str(params["entry_end"]).split(":"))
    start_value = start_hour * 60 + start_minute
    end_value = end_hour * 60 + end_minute
    minute_value = _minute_value(ts)
    return start_value <= minute_value < end_value


def _reclaim_level_not_touched_yet(
    frame: pd.DataFrame,
    *,
    start_i: int,
    end_i: int,
    direction: str,
    price: float,
) -> bool:
    if end_i < start_i:
        return True
    window = frame.iloc[start_i : end_i + 1]
    if window.empty:
        return True
    if direction == "short":
        return float(window["high"].max()) < price
    return float(window["low"].min()) > price


def _build_signal(
    *,
    direction: str,
    entry_price: float,
    stop_price: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "direction": "short" if direction == "short" else "long",
        "entry_mode": "limit",
        "limit_price": entry_price,
        "stop_mode": "price",
        "stop_price": stop_price,
        "target_mode": "rr",
        "target_rr": float(params["target_rr"]),
        "cooldown_bars": int(params.get("cooldown_bars", 0)),
        "session_name": str(params["session_name"]),
    }


def _find_recent_failure(
    frame: pd.DataFrame,
    i: int,
    params: dict[str, Any],
    *,
    direction: str,
) -> dict[str, float] | None:
    failure_col = "odr_failure_short" if direction == "short" else "odr_failure_long"
    reclaim_col = "odr_reclaim_level_short" if direction == "short" else "odr_reclaim_level_long"
    drive_extreme_col = "odr_drive_high" if direction == "short" else "odr_drive_low"

    search_start = max(0, i - int(params["retest_window_bars"]))
    current_close = float(frame["close"].iat[i])
    stop_buffer = float(params["stop_buffer_pips"]) * PIP_SIZE

    for k in range(i, search_start - 1, -1):
        if not bool(frame[failure_col].iat[k]):
            continue

        entry_price = float(frame[reclaim_col].iat[k])
        drive_extreme = float(frame[drive_extreme_col].iat[k])
        if not finite(entry_price) or not finite(drive_extreme):
            continue

        if direction == "short":
            stop_price = drive_extreme + stop_buffer
            if not (entry_price > current_close and stop_price > entry_price):
                continue
        else:
            stop_price = drive_extreme - stop_buffer
            if not (entry_price < current_close and stop_price < entry_price):
                continue

        if not _reclaim_level_not_touched_yet(
            frame,
            start_i=k + 1,
            end_i=i,
            direction=direction,
            price=entry_price,
        ):
            continue

        return {"entry_price": entry_price, "stop_price": stop_price}
    return None


def signal(frame: pd.DataFrame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
    ts = frame.index[i]
    if i < 2 or not is_in_session(ts, str(params["session_name"])) or not _is_trade_window(ts, params):
        return None

    short_setup = _find_recent_failure(frame, i, params, direction="short")
    if short_setup is not None:
        return _build_signal(direction="short", params=params, **short_setup)

    long_setup = _find_recent_failure(frame, i, params, direction="long")
    if long_setup is not None:
        return _build_signal(direction="long", params=params, **long_setup)

    return None
