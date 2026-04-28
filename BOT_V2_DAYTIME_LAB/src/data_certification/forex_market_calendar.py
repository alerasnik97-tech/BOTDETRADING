from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class TimeWindow:
    start_hour: float
    end_hour: float


USER_WINDOW = TimeWindow(7.0, 20.0)
PHASE19_WINDOW = TimeWindow(8.0, 16.5)
PHASE18_WINDOW = TimeWindow(8.0, 11.0)
ROLLOVER_WINDOW = TimeWindow(17.0, 19.0)


def to_utc(timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_ny(timestamp) -> pd.Timestamp:
    return to_utc(timestamp).tz_convert(NY_TZ)


def hour_decimal(timestamp) -> float:
    ts = to_ny(timestamp)
    return ts.hour + ts.minute / 60.0 + ts.second / 3600.0


def is_market_expected_open(timestamp_utc) -> bool:
    ts = to_ny(timestamp_utc)
    dow = ts.dayofweek
    hour = hour_decimal(ts)
    if dow == 5:
        return False
    if dow == 6:
        return hour >= 17.0
    if dow == 4:
        return hour < 17.0
    return True


def is_within_user_window(timestamp_utc, start_hour: float = 7.0, end_hour: float = 20.0) -> bool:
    hour = hour_decimal(timestamp_utc)
    return start_hour <= hour < end_hour


def is_within_phase19_candidate_window(timestamp_utc, start_hour: float = 8.0, end_hour: float = 16.5) -> bool:
    hour = hour_decimal(timestamp_utc)
    return start_hour <= hour < end_hour


def is_within_phase18_window(timestamp_utc, start_hour: float = 8.0, end_hour: float = 11.0) -> bool:
    hour = hour_decimal(timestamp_utc)
    return start_hour <= hour < end_hour


def is_rollover(timestamp_utc, start_hour: float = 17.0, end_hour: float = 19.0) -> bool:
    hour = hour_decimal(timestamp_utc)
    return start_hour <= hour < end_hour


def session_label(timestamp_utc) -> str:
    ts = to_ny(timestamp_utc)
    if not is_market_expected_open(timestamp_utc):
        return "MARKET_CLOSED"
    if is_rollover(timestamp_utc):
        return "ROLLOVER"
    if is_within_phase18_window(timestamp_utc):
        return "PHASE18_0800_1100"
    if is_within_phase19_candidate_window(timestamp_utc):
        return "PHASE19_0800_1630"
    if is_within_user_window(timestamp_utc):
        return "USER_WINDOW_OUTSIDE_PHASE19"
    return f"OUTSIDE_USER_WINDOW_{ts.strftime('%H%M')}"


def missing_bar_times(start_utc, end_utc, freq_minutes: int = 3) -> pd.DatetimeIndex:
    start = to_utc(start_utc)
    end = to_utc(end_utc)
    if end < start:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start, end, freq=f"{freq_minutes}min", tz="UTC")


def interval_any(times: pd.DatetimeIndex, predicate) -> bool:
    if len(times) == 0:
        return False
    return any(bool(predicate(ts)) for ts in times)


def interval_market_expected_open(times: pd.DatetimeIndex) -> bool:
    return interval_any(times, is_market_expected_open)


def interval_in_user_window(times: pd.DatetimeIndex) -> bool:
    return interval_any(times, is_within_user_window)


def interval_in_phase19_window(times: pd.DatetimeIndex) -> bool:
    return interval_any(times, is_within_phase19_candidate_window)


def interval_in_phase18_window(times: pd.DatetimeIndex) -> bool:
    return interval_any(times, is_within_phase18_window)


def interval_in_rollover(times: pd.DatetimeIndex) -> bool:
    return interval_any(times, is_rollover)
