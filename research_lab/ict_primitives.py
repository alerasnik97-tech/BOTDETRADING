from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from research_lab.data_loader import fixed_session_window_components, fx_session_date


DEFAULT_PIP_SIZE = 0.0001
DEFAULT_SESSION_WINDOWS: dict[str, tuple[str, str]] = {
    "asia": ("19:00", "03:00"),
    "london": ("03:00", "07:00"),
    "ny_opening": ("08:00", "09:30"),
}

DEFAULT_SHORT_LIQUIDITY_LEVELS = ("prev_month_high", "prev_week_high", "prev_day_high", "london_high", "asia_high")
DEFAULT_LONG_LIQUIDITY_LEVELS = ("prev_month_low", "prev_week_low", "prev_day_low", "london_low", "asia_low")



@dataclass(frozen=True)
class SweepEvent:
    direction: str
    level_name: str
    level_price: float
    sweep_price: float
    reclaim_price: float
    bar_index: int


@dataclass(frozen=True)
class IfvgEvent:
    direction: str
    source_fvg_direction: str
    source_bar_index: int
    inversion_bar_index: int
    bottom: float
    top: float
    midpoint: float
    size_pips: float
    size_atr: float


def _minute_value(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def _close_location(frame: pd.DataFrame) -> pd.Series:
    bar_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    close_location = (frame["close"] - frame["low"]) / bar_range
    return close_location.fillna(0.5)


def ensure_session_range_columns(frame: pd.DataFrame, label: str, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    result = frame.copy()
    in_window, session_dates, complete_mask = fixed_session_window_components(result.index, start_hhmm, end_hhmm)

    sliced = result.loc[in_window]
    if sliced.empty:
        result[f"{label}_high"] = np.nan
        result[f"{label}_low"] = np.nan
        result[f"{label}_complete"] = False
        return result

    levels = sliced.groupby(session_dates.loc[sliced.index]).agg({"high": "max", "low": "min"})
    result[f"{label}_high"] = session_dates.map(levels["high"])
    result[f"{label}_low"] = session_dates.map(levels["low"])
    result[f"{label}_complete"] = complete_mask & result[f"{label}_high"].notna()
    result.loc[~result[f"{label}_complete"], [f"{label}_high", f"{label}_low"]] = np.nan
    return result


def add_session_level_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "asia": ("session_range_high_19_00_03_00", "session_range_low_19_00_03_00", "session_range_complete_19_00_03_00"),
        "london": ("session_range_high_03_00_07_00", "session_range_low_03_00_07_00", "session_range_complete_03_00_07_00"),
    }
    for label, (high_col, low_col, complete_col) in aliases.items():
        if all(column in result.columns for column in (high_col, low_col, complete_col)):
            result[f"{label}_high"] = result[high_col]
            result[f"{label}_low"] = result[low_col]
            result[f"{label}_complete"] = result[complete_col].astype(bool)
        else:
            start_hhmm, end_hhmm = DEFAULT_SESSION_WINDOWS[label]
            result = ensure_session_range_columns(result, label, start_hhmm, end_hhmm)

    result = ensure_session_range_columns(result, "ny_opening", *DEFAULT_SESSION_WINDOWS["ny_opening"])
    return result


def add_previous_period_levels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    session_dates = pd.Series(pd.to_datetime(fx_session_date(result.index)), index=result.index)

    if "prev_day_high" not in result.columns or "prev_day_low" not in result.columns:
        day_levels = (
            pd.DataFrame({"session_date": session_dates, "high": result["high"], "low": result["low"]})
            .groupby("session_date")
            .agg(day_high=("high", "max"), day_low=("low", "min"))
        )
        day_levels["prev_day_high"] = day_levels["day_high"].shift(1)
        day_levels["prev_day_low"] = day_levels["day_low"].shift(1)
        result["prev_day_high"] = session_dates.map(day_levels["prev_day_high"])
        result["prev_day_low"] = session_dates.map(day_levels["prev_day_low"])

    week_key = session_dates.dt.to_period("W-FRI")
    week_levels = (
        pd.DataFrame({"week_key": week_key, "high": result["high"], "low": result["low"], "open": result["open"]})
        .groupby("week_key")
        .agg(week_high=("high", "max"), week_low=("low", "min"), week_open=("open", "first"))
    )
    week_levels["prev_week_high"] = week_levels["week_high"].shift(1)
    week_levels["prev_week_low"] = week_levels["week_low"].shift(1)
    week_levels["weekly_open"] = week_levels["week_open"] # Current week open
    
    result["prev_week_high"] = week_key.map(week_levels["prev_week_high"])
    result["prev_week_low"] = week_key.map(week_levels["prev_week_low"])
    result["weekly_open"] = week_key.map(week_levels["weekly_open"])

    month_key = session_dates.dt.to_period("M")
    month_levels = (
        pd.DataFrame({"month_key": month_key, "high": result["high"], "low": result["low"]})
        .groupby("month_key")
        .agg(month_high=("high", "max"), month_low=("low", "min"))
    )
    month_levels["prev_month_high"] = month_levels["month_high"].shift(1)
    month_levels["prev_month_low"] = month_levels["month_low"].shift(1)
    result["prev_month_high"] = month_key.map(month_levels["prev_month_high"])
    result["prev_month_low"] = month_key.map(month_levels["prev_month_low"])
    return result



def add_displacement_metrics(frame: pd.DataFrame, recent_lookback: int = 20) -> pd.DataFrame:
    result = frame.copy()
    bar_range = (result["high"] - result["low"]).replace(0.0, np.nan)
    result["close_location"] = _close_location(result)
    result["body_fraction"] = (result["close"] - result["open"]).abs() / bar_range
    result["body_to_atr"] = (result["close"] - result["open"]).abs() / result["atr14"].replace(0.0, np.nan)
    result["range_to_atr"] = bar_range / result["atr14"].replace(0.0, np.nan)
    recent_avg_range = bar_range.shift(1).rolling(recent_lookback).mean()
    result["range_vs_recent"] = bar_range / recent_avg_range.replace(0.0, np.nan)
    return result


def bullish_displacement(
    frame: pd.DataFrame,
    i: int,
    *,
    min_body_atr: float,
    min_body_fraction: float,
    min_close_location: float,
    min_range_expansion: float,
) -> bool:
    return bool(
        float(frame["close"].iat[i]) > float(frame["open"].iat[i])
        and float(frame["body_to_atr"].iat[i]) >= min_body_atr
        and float(frame["body_fraction"].iat[i]) >= min_body_fraction
        and float(frame["close_location"].iat[i]) >= min_close_location
        and float(frame["range_vs_recent"].iat[i]) >= min_range_expansion
    )


def bearish_displacement(
    frame: pd.DataFrame,
    i: int,
    *,
    min_body_atr: float,
    min_body_fraction: float,
    max_close_location: float,
    min_range_expansion: float,
) -> bool:
    return bool(
        float(frame["close"].iat[i]) < float(frame["open"].iat[i])
        and float(frame["body_to_atr"].iat[i]) >= min_body_atr
        and float(frame["body_fraction"].iat[i]) >= min_body_fraction
        and float(frame["close_location"].iat[i]) <= max_close_location
        and float(frame["range_vs_recent"].iat[i]) >= min_range_expansion
    )


def add_fvg_columns(frame: pd.DataFrame, pip_size: float = DEFAULT_PIP_SIZE) -> pd.DataFrame:
    result = frame.copy()
    result["bullish_fvg"] = False
    result["bearish_fvg"] = False
    result["bullish_fvg_bottom"] = np.nan
    result["bullish_fvg_top"] = np.nan
    result["bullish_fvg_mid"] = np.nan
    result["bullish_fvg_size_pips"] = np.nan
    result["bullish_fvg_size_atr"] = np.nan
    result["bearish_fvg_bottom"] = np.nan
    result["bearish_fvg_top"] = np.nan
    result["bearish_fvg_mid"] = np.nan
    result["bearish_fvg_size_pips"] = np.nan
    result["bearish_fvg_size_atr"] = np.nan

    if len(result) < 3:
        return result

    bullish_mask = result["low"] > result["high"].shift(2)
    bearish_mask = result["high"] < result["low"].shift(2)

    bullish_bottom = result["high"].shift(2)
    bullish_top = result["low"]
    bearish_bottom = result["high"]
    bearish_top = result["low"].shift(2)

    bullish_size = bullish_top - bullish_bottom
    bearish_size = bearish_top - bearish_bottom

    result.loc[bullish_mask, "bullish_fvg"] = True
    result.loc[bullish_mask, "bullish_fvg_bottom"] = bullish_bottom.loc[bullish_mask]
    result.loc[bullish_mask, "bullish_fvg_top"] = bullish_top.loc[bullish_mask]
    result.loc[bullish_mask, "bullish_fvg_mid"] = (bullish_bottom.loc[bullish_mask] + bullish_top.loc[bullish_mask]) / 2.0
    result.loc[bullish_mask, "bullish_fvg_size_pips"] = bullish_size.loc[bullish_mask] / pip_size
    result.loc[bullish_mask, "bullish_fvg_size_atr"] = bullish_size.loc[bullish_mask] / result["atr14"].replace(0.0, np.nan).loc[bullish_mask]

    result.loc[bearish_mask, "bearish_fvg"] = True
    result.loc[bearish_mask, "bearish_fvg_bottom"] = bearish_bottom.loc[bearish_mask]
    result.loc[bearish_mask, "bearish_fvg_top"] = bearish_top.loc[bearish_mask]
    result.loc[bearish_mask, "bearish_fvg_mid"] = (bearish_bottom.loc[bearish_mask] + bearish_top.loc[bearish_mask]) / 2.0
    result.loc[bearish_mask, "bearish_fvg_size_pips"] = bearish_size.loc[bearish_mask] / pip_size
    result.loc[bearish_mask, "bearish_fvg_size_atr"] = bearish_size.loc[bearish_mask] / result["atr14"].replace(0.0, np.nan).loc[bearish_mask]
    return result


def add_pivot_structure_columns(
    frame: pd.DataFrame,
    *,
    left_bars: int = 2,
    right_bars: int = 2,
    break_buffer_pips: float = 0.5,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> pd.DataFrame:
    result = frame.copy()
    window = left_bars + right_bars + 1

    high_roll = result["high"].rolling(window=window, center=True)
    low_roll = result["low"].rolling(window=window, center=True)
    pivot_high_raw = result["high"].eq(high_roll.max()) & high_roll.apply(lambda values: int(np.sum(values == np.max(values))) == 1, raw=True).fillna(0).astype(bool)
    pivot_low_raw = result["low"].eq(low_roll.min()) & low_roll.apply(lambda values: int(np.sum(values == np.min(values))) == 1, raw=True).fillna(0).astype(bool)

    result["pivot_high_raw"] = pivot_high_raw.fillna(False)
    result["pivot_low_raw"] = pivot_low_raw.fillna(False)
    result["confirmed_pivot_high"] = result["high"].where(result["pivot_high_raw"]).shift(right_bars)
    result["confirmed_pivot_low"] = result["low"].where(result["pivot_low_raw"]).shift(right_bars)

    last_swing_high: list[float] = []
    last_swing_low: list[float] = []
    structure_bias: list[int] = []
    bullish_break_close: list[bool] = []
    bearish_break_close: list[bool] = []
    bullish_bos: list[bool] = []
    bearish_bos: list[bool] = []
    bullish_choch: list[bool] = []
    bearish_choch: list[bool] = []

    current_high = np.nan
    prev_high = np.nan
    current_low = np.nan
    prev_low = np.nan
    bias = 0
    high_broken = False
    low_broken = False
    break_buffer = break_buffer_pips * pip_size

    for i in range(len(result)):
        confirmed_high = result["confirmed_pivot_high"].iat[i]
        confirmed_low = result["confirmed_pivot_low"].iat[i]
        if _finite(confirmed_high):
            prev_high = current_high
            current_high = float(confirmed_high)
            high_broken = False
        if _finite(confirmed_low):
            prev_low = current_low
            current_low = float(confirmed_low)
            low_broken = False

        if _finite(prev_high) and _finite(prev_low) and _finite(current_high) and _finite(current_low):
            if current_high > prev_high and current_low > prev_low:
                bias = 1
            elif current_high < prev_high and current_low < prev_low:
                bias = -1

        close_price = float(result["close"].iat[i])
        bull_break = _finite(current_high) and (close_price > float(current_high) + break_buffer) and not high_broken
        bear_break = _finite(current_low) and (close_price < float(current_low) - break_buffer) and not low_broken

        bull_bos = bull_break and bias > 0
        bear_bos = bear_break and bias < 0
        bull_choch = bull_break and bias <= 0
        bear_choch = bear_break and bias >= 0

        if bull_break:
            high_broken = True
            bias = 1
        if bear_break:
            low_broken = True
            bias = -1

        last_swing_high.append(float(current_high) if _finite(current_high) else np.nan)
        last_swing_low.append(float(current_low) if _finite(current_low) else np.nan)
        structure_bias.append(int(bias))
        bullish_break_close.append(bool(bull_break))
        bearish_break_close.append(bool(bear_break))
        bullish_bos.append(bool(bull_bos))
        bearish_bos.append(bool(bear_bos))
        bullish_choch.append(bool(bull_choch))
        bearish_choch.append(bool(bear_choch))

    result["last_confirmed_swing_high"] = last_swing_high
    result["last_confirmed_swing_low"] = last_swing_low
    result["structure_bias"] = structure_bias
    result["bullish_break_close"] = bullish_break_close
    result["bearish_break_close"] = bearish_break_close
    result["bullish_bos"] = bullish_bos
    result["bearish_bos"] = bearish_bos
    result["bullish_choch"] = bullish_choch
    result["bearish_choch"] = bearish_choch
    return result


def add_equal_high_low_columns(
    frame: pd.DataFrame,
    *,
    tolerance_pips: float = 1.0,
    max_separation_bars: int = 60,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> pd.DataFrame:
    result = frame.copy()
    result["equal_high"] = False
    result["equal_low"] = False
    result["equal_high_level"] = np.nan
    result["equal_low_level"] = np.nan

    tolerance = tolerance_pips * pip_size
    last_high_price = np.nan
    last_high_index = -10_000
    last_low_price = np.nan
    last_low_index = -10_000

    for i in range(len(result)):
        confirmed_high = result["confirmed_pivot_high"].iat[i] if "confirmed_pivot_high" in result.columns else np.nan
        confirmed_low = result["confirmed_pivot_low"].iat[i] if "confirmed_pivot_low" in result.columns else np.nan

        if _finite(confirmed_high):
            high_price = float(confirmed_high)
            if _finite(last_high_price) and abs(high_price - float(last_high_price)) <= tolerance and (i - last_high_index) <= max_separation_bars:
                result.iat[i, result.columns.get_loc("equal_high")] = True
                result.iat[i, result.columns.get_loc("equal_high_level")] = (high_price + float(last_high_price)) / 2.0
            last_high_price = high_price
            last_high_index = i

        if _finite(confirmed_low):
            low_price = float(confirmed_low)
            if _finite(last_low_price) and abs(low_price - float(last_low_price)) <= tolerance and (i - last_low_index) <= max_separation_bars:
                result.iat[i, result.columns.get_loc("equal_low")] = True
                result.iat[i, result.columns.get_loc("equal_low_level")] = (low_price + float(last_low_price)) / 2.0
            last_low_price = low_price
            last_low_index = i

    return result


def add_premium_discount_columns(frame: pd.DataFrame, high_col: str, low_col: str, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    midpoint_col = f"{prefix}_midpoint"
    result[midpoint_col] = (result[high_col] + result[low_col]) / 2.0
    result[f"close_in_discount_{prefix}"] = result["close"] <= result[midpoint_col]
    result[f"close_in_premium_{prefix}"] = result["close"] >= result[midpoint_col]
    result[f"wick_in_discount_{prefix}"] = result["low"] <= result[midpoint_col]
    result[f"wick_in_premium_{prefix}"] = result["high"] >= result[midpoint_col]
    return result


def add_ict_primitives(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = add_session_level_aliases(result)
    result = add_previous_period_levels(result)
    result = add_displacement_metrics(result)
    result = add_fvg_columns(result)
    result = add_pivot_structure_columns(result)
    result = add_equal_high_low_columns(result)
    if "prev_day_high" in result.columns and "prev_day_low" in result.columns:
        result = add_premium_discount_columns(result, "prev_day_high", "prev_day_low", "prev_day")
    if "asia_high" in result.columns and "asia_low" in result.columns:
        result = add_premium_discount_columns(result, "asia_high", "asia_low", "asia")
    if "london_high" in result.columns and "london_low" in result.columns:
        result = add_premium_discount_columns(result, "london_high", "london_low", "london")
    return result


def _bar_sweeps_above(frame: pd.DataFrame, i: int, level_price: float, min_penetration_pips: float, pip_size: float) -> bool:
    return bool(
        _finite(level_price)
        and float(frame["high"].iat[i]) >= float(level_price) + (min_penetration_pips * pip_size)
        and float(frame["close"].iat[i]) < float(level_price)
    )


def _bar_sweeps_below(frame: pd.DataFrame, i: int, level_price: float, min_penetration_pips: float, pip_size: float) -> bool:
    return bool(
        _finite(level_price)
        and float(frame["low"].iat[i]) <= float(level_price) - (min_penetration_pips * pip_size)
        and float(frame["close"].iat[i]) > float(level_price)
    )


def find_recent_sweep_event(
    frame: pd.DataFrame,
    i: int,
    *,
    direction: str,
    min_penetration_pips: float,
    max_age_bars: int = 0,
    level_columns: Iterable[str] | None = None,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> SweepEvent | None:
    if direction == "short":
        candidate_levels = tuple(level_columns or DEFAULT_SHORT_LIQUIDITY_LEVELS)
    else:
        candidate_levels = tuple(level_columns or DEFAULT_LONG_LIQUIDITY_LEVELS)

    start = max(0, i - int(max_age_bars))
    for j in range(i, start - 1, -1):
        for level_name in candidate_levels:
            if level_name not in frame.columns:
                continue
            level_price = frame[level_name].iat[j]
            if not _finite(level_price):
                continue
            if direction == "short" and _bar_sweeps_above(frame, j, float(level_price), min_penetration_pips, pip_size):
                return SweepEvent(
                    direction="short",
                    level_name=level_name,
                    level_price=float(level_price),
                    sweep_price=float(frame["high"].iat[j]),
                    reclaim_price=float(frame["close"].iat[j]),
                    bar_index=j,
                )
            if direction == "long" and _bar_sweeps_below(frame, j, float(level_price), min_penetration_pips, pip_size):
                return SweepEvent(
                    direction="long",
                    level_name=level_name,
                    level_price=float(level_price),
                    sweep_price=float(frame["low"].iat[j]),
                    reclaim_price=float(frame["close"].iat[j]),
                    bar_index=j,
                )
    return None


def find_recent_ifvg_event(
    frame: pd.DataFrame,
    i: int,
    *,
    direction: str,
    min_fvg_pips: float,
    min_fvg_atr: float,
    max_fvg_age_bars: int,
    max_inversion_bars: int,
    max_retest_bars: int,
    require_break_close: bool = True,
) -> IfvgEvent | None:
    if i < 3:
        return None

    direction_normalized = str(direction).strip().lower()
    if direction_normalized not in {"long", "short"}:
        raise ValueError(f"direction invalida para IFVG: {direction!r}")

    start = max(2, i - int(max_fvg_age_bars))
    for source_i in range(i - 1, start - 1, -1):
        if direction_normalized == "long":
            is_candidate = bool(frame["bearish_fvg"].iat[source_i])
            bottom = float(frame["bearish_fvg_bottom"].iat[source_i])
            top = float(frame["bearish_fvg_top"].iat[source_i])
            size_pips = float(frame["bearish_fvg_size_pips"].iat[source_i])
            size_atr = float(frame["bearish_fvg_size_atr"].iat[source_i])
            break_col = "bullish_break_close"
            source_side = "bearish"
        else:
            is_candidate = bool(frame["bullish_fvg"].iat[source_i])
            bottom = float(frame["bullish_fvg_bottom"].iat[source_i])
            top = float(frame["bullish_fvg_top"].iat[source_i])
            size_pips = float(frame["bullish_fvg_size_pips"].iat[source_i])
            size_atr = float(frame["bullish_fvg_size_atr"].iat[source_i])
            break_col = "bearish_break_close"
            source_side = "bullish"

        if not is_candidate or not _finite(bottom) or not _finite(top):
            continue
        if not _finite(size_pips) or not _finite(size_atr):
            continue
        if size_pips < float(min_fvg_pips) or size_atr < float(min_fvg_atr):
            continue

        inversion_deadline = min(i, source_i + int(max_inversion_bars))
        inversion_i: int | None = None
        for candidate_i in range(source_i + 1, inversion_deadline + 1):
            close_price = float(frame["close"].iat[candidate_i])
            break_close = bool(frame[break_col].iat[candidate_i]) if break_col in frame.columns else True
            if direction_normalized == "long":
                if close_price > top and (not require_break_close or break_close):
                    inversion_i = candidate_i
                    break
            else:
                if close_price < bottom and (not require_break_close or break_close):
                    inversion_i = candidate_i
                    break

        if inversion_i is None:
            continue
        if (i - inversion_i) > int(max_retest_bars):
            continue

        invalidated = False
        for check_i in range(inversion_i + 1, i + 1):
            close_price = float(frame["close"].iat[check_i])
            if direction_normalized == "long" and close_price < bottom:
                invalidated = True
                break
            if direction_normalized == "short" and close_price > top:
                invalidated = True
                break
        if invalidated:
            continue

        return IfvgEvent(
            direction=direction_normalized,
            source_fvg_direction=source_side,
            source_bar_index=source_i,
            inversion_bar_index=inversion_i,
            bottom=bottom,
            top=top,
            midpoint=(bottom + top) / 2.0,
            size_pips=size_pips,
            size_atr=size_atr,
        )
    return None


def passes_h1_ema_bias(frame: pd.DataFrame, i: int, direction: str) -> bool:
    if direction == "long":
        return bool(
            float(frame["h1_ema50"].iat[i]) > float(frame["h1_ema200"].iat[i])
            and float(frame["h1_ema200_slope_5"].iat[i]) > 0
        )
    return bool(
        float(frame["h1_ema50"].iat[i]) < float(frame["h1_ema200"].iat[i])
        and float(frame["h1_ema200_slope_5"].iat[i]) < 0
    )


def passes_prev_day_premium_discount(frame: pd.DataFrame, i: int, direction: str) -> bool:
    if "prev_day_midpoint" not in frame.columns:
        return True
    midpoint = frame["prev_day_midpoint"].iat[i]
    if not _finite(midpoint):
        return False
    if direction == "long":
        return bool(frame["close_in_discount_prev_day"].iat[i])
    return bool(frame["close_in_premium_prev_day"].iat[i])
