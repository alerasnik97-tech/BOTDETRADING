from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab.build_am_grade_news_dataset import DEFAULT_OUTPUT_FILE as AM_NEWS_FILE
from research_lab.build_am_grade_news_dataset import build_am_grade_news_dataset
from research_lab.config import (
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NY_TZ,
    NewsConfig,
    with_execution_mode,
)
from research_lab.data_loader import (
    fx_market_mask,
    load_high_precision_package,
    prepare_common_frame,
    resample_ohlcv_to_timeframe,
    validate_price_frame,
)
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import export_strategy_bundle, summarize_result
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module


RESULTS_DIR = Path("results") / strategy_module.NAME
PAIR = "EURUSD"
TIMEFRAME = "M3"
CONTEXT_TIMEFRAME = "M5"
OPERATIVE_START_MINUTE = 7 * 60
SWEEP_LATEST_MINUTE = 10 * 60 + 30
NON_TRADABLE_FROM_MINUTE = 10 * 60 + 45
FORCE_CLOSE = "11:30"
PIP_SIZE = 0.0001
MIN_SWEEP_PIPS = 1.0
MAX_SWEEP_ATR_MULT = 0.25
MAX_CONFIRM_BARS = 6
STOP_BUFFER_PIPS = 1.0
TARGET_RR = 2.1
BREAK_EVEN_AT_R = 1.2
MAX_HOLD_BARS = 21

PERIODS: dict[str, tuple[str, str]] = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}

SERIOUS_DEV_MIN_TRADES = 48
SERIOUS_DEV_MIN_PF = 1.10
SERIOUS_DEV_MIN_EXPECTANCY = 0.03
SERIOUS_MIN_VAL_HOLD_TRADES = 12
SERIOUS_MIN_VAL_HOLD_PF = 0.95
SERIOUS_MIN_VAL_HOLD_EXPECTANCY = 0.0

# Internal defaults (can be overridden by audit scripts)
DEFAULT_SHORT_LEVELS: tuple[tuple[str, str], ...] = (
    ("prev_day_high", "prev_day"),
    ("asia_high", "asia"),
    ("london_high", "london"),
)
DEFAULT_LONG_LEVELS: tuple[tuple[str, str], ...] = (
    ("prev_day_low", "prev_day"),
    ("asia_low", "asia"),
    ("london_low", "london"),
)



def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_{strategy_module.NAME}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_engine_config() -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        max_spread_pips=2.0,
        slippage_pips=0.1,
        commission_per_lot_roundturn_usd=7.0,
        max_trades_per_day=1,
        session_cutoff=FORCE_CLOSE,
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, "high_precision_mode")


def build_news_config() -> NewsConfig:
    return NewsConfig(
        enabled=True,
        file_path=AM_NEWS_FILE,
        raw_file_path=AM_NEWS_FILE,
        source_approved=True,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
        currencies=("USD", "EUR"),
        impact_levels=("HIGH",),
    )


def schedule_used() -> dict[str, str]:
    return {
        "entry_start": "07:00",
        "entry_end": "11:00",
        "force_close": FORCE_CLOSE,
        "sweep_frame": CONTEXT_TIMEFRAME,
        "confirmation_frame": TIMEFRAME,
        "asia_window": "19:00-03:00",
        "london_window": "03:00-07:00",
    }


def _minute_value(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _hhmm(minute_value: int) -> str:
    hour = int(minute_value) // 60
    minute = int(minute_value) % 60
    return f"{hour:02d}:{minute:02d}"


def _finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def _profit_factor(pnl_r: pd.Series) -> float:
    gross_profit = float(pnl_r[pnl_r > 0].sum())
    gross_loss = float(pnl_r[pnl_r < 0].sum())
    return gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")


def _filtered_high_precision_package(start: str, end: str) -> dict[str, pd.DataFrame]:
    package = load_high_precision_package(PAIR, DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    filtered: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame[fx_market_mask(frame.index)].copy()
        validate_price_frame(frame)
        filtered[f"{side}_m1"] = frame
    return filtered


def _build_m3_m5_frames(mid_m1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m3_frame = prepare_common_frame(mid_m1, target_timeframe=TIMEFRAME)
    m5_frame = prepare_common_frame(mid_m1, target_timeframe=CONTEXT_TIMEFRAME)
    return m3_frame, m5_frame


def _align_precision_package(filtered: dict[str, pd.DataFrame], frame_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    bid_exec = resample_ohlcv_to_timeframe(filtered["bid_m1"], TIMEFRAME).loc[frame_index].copy()
    ask_exec = resample_ohlcv_to_timeframe(filtered["ask_m1"], TIMEFRAME).loc[frame_index].copy()
    mid_exec = resample_ohlcv_to_timeframe(filtered["mid_m1"], TIMEFRAME).loc[frame_index].copy()
    return {
        "bid_m1": filtered["bid_m1"].copy(),
        "ask_m1": filtered["ask_m1"].copy(),
        "mid_m1": filtered["mid_m1"].copy(),
        "bid_exec": bid_exec.copy(),
        "ask_exec": ask_exec.copy(),
        "mid_exec": mid_exec.copy(),
        "bid_m15": bid_exec.copy(),
        "ask_m15": ask_exec.copy(),
        "mid_m15": mid_exec.copy(),
    }


def _build_trade_window_contract(
    index: pd.DatetimeIndex,
    news_events: pd.DataFrame,
    news_config: NewsConfig,
) -> pd.DataFrame:
    result = pd.DataFrame(index=index)
    result["trade_day"] = index.date
    result["trade_minute"] = (index.hour * 60 + index.minute).astype(int)
    result["els_tradable_from_minute"] = OPERATIVE_START_MINUTE
    result["els_day_consumed"] = False
    result["els_last_block_event_name"] = ""
    result["els_last_block_event_time_ny"] = ""

    if news_events is not None and not news_events.empty:
        events = news_events.copy()
        event_times = pd.to_datetime(events["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
        event_minutes = event_times.dt.hour * 60 + event_times.dt.minute
        session_mask = (event_minutes >= OPERATIVE_START_MINUTE) & (event_minutes <= (10 * 60))
        relevant = pd.DataFrame(
            {
                "trade_day": event_times.dt.date,
                "event_time_ny": event_times,
                "event_name_normalized": events["event_name_normalized"].astype(str),
                "block_end_minute": event_minutes + int(news_config.post_minutes),
            }
        ).loc[session_mask].copy()
        if not relevant.empty:
            latest_rows = relevant.sort_values(["trade_day", "block_end_minute", "event_time_ny"]).groupby("trade_day").tail(1)
            tradable_from_map = latest_rows.set_index("trade_day")["block_end_minute"]
            name_map = latest_rows.set_index("trade_day")["event_name_normalized"]
            time_map = latest_rows.set_index("trade_day")["event_time_ny"].astype(str)
            result["els_tradable_from_minute"] = np.maximum(
                OPERATIVE_START_MINUTE,
                result["trade_day"].map(tradable_from_map).fillna(OPERATIVE_START_MINUTE).astype(int),
            )
            result["els_last_block_event_name"] = result["trade_day"].map(name_map).fillna("")
            result["els_last_block_event_time_ny"] = result["trade_day"].map(time_map).fillna("")

    result["els_day_consumed"] = result["els_tradable_from_minute"] >= NON_TRADABLE_FROM_MINUTE
    result["els_tradable_from_hhmm"] = result["els_tradable_from_minute"].map(_hhmm)
    result["els_trade_window_open"] = (
        (result["trade_minute"] >= result["els_tradable_from_minute"])
        & (result["trade_minute"] < 11 * 60)
        & (~result["els_day_consumed"])
    )
    return result.drop(columns=["trade_minute"])


def _directional_sweep_candidate(row: pd.Series, *, direction: str, short_levels: tuple[tuple[str, str], ...] | None = None, long_levels: tuple[tuple[str, str], ...] | None = None) -> tuple[str, str, float, float, float] | None:
    level_specs = short_levels if direction == "short" else long_levels
    if level_specs is None:
        level_specs = DEFAULT_SHORT_LEVELS if direction == "short" else DEFAULT_LONG_LEVELS
        
    for level_name, source_kind in level_specs:

        if level_name not in row.index:
            continue
        level_price = row.get(level_name)
        atr_value = row.get("atr14")
        if not _finite(level_price) or not _finite(atr_value) or float(atr_value) <= 0:
            continue
        level_price_f = float(level_price)
        atr_value_f = float(atr_value)
        if direction == "short":
            overshoot = float(row["high"]) - level_price_f
            if float(row["high"]) >= level_price_f + (MIN_SWEEP_PIPS * PIP_SIZE) and overshoot <= MAX_SWEEP_ATR_MULT * atr_value_f and float(row["close"]) <= level_price_f:
                return level_name, source_kind, level_price_f, float(row["high"]), float(row["low"])
        else:
            overshoot = level_price_f - float(row["low"])
            if float(row["low"]) <= level_price_f - (MIN_SWEEP_PIPS * PIP_SIZE) and overshoot <= MAX_SWEEP_ATR_MULT * atr_value_f and float(row["close"]) >= level_price_f:
                return level_name, source_kind, level_price_f, float(row["low"]), float(row["high"])
    return None


def _bar_confirmation(frame: pd.DataFrame, i: int, *, direction: str, sweep_extreme_opposite: float) -> bool:
    if direction == "short":
        return bool(
            float(frame["close"].iat[i]) < float(frame["open"].iat[i])
            and _finite(frame["body_to_atr"].iat[i]) and float(frame["body_to_atr"].iat[i]) >= 0.25
            and _finite(frame["body_fraction"].iat[i]) and float(frame["body_fraction"].iat[i]) >= 0.45
            and _finite(frame["range_vs_recent"].iat[i]) and float(frame["range_vs_recent"].iat[i]) >= 1.0
            and _finite(frame["close_location"].iat[i]) and float(frame["close_location"].iat[i]) <= 0.35
            and bool(frame["bearish_break_close"].iat[i])
            and _finite(frame["last_confirmed_swing_low"].iat[i])
            and float(frame["close"].iat[i]) < float(frame["last_confirmed_swing_low"].iat[i])
            and float(frame["close"].iat[i]) < sweep_extreme_opposite
        )
    return bool(
        float(frame["close"].iat[i]) > float(frame["open"].iat[i])
        and _finite(frame["body_to_atr"].iat[i]) and float(frame["body_to_atr"].iat[i]) >= 0.25
        and _finite(frame["body_fraction"].iat[i]) and float(frame["body_fraction"].iat[i]) >= 0.45
        and _finite(frame["range_vs_recent"].iat[i]) and float(frame["range_vs_recent"].iat[i]) >= 1.0
        and _finite(frame["close_location"].iat[i]) and float(frame["close_location"].iat[i]) >= 0.65
        and bool(frame["bullish_break_close"].iat[i])
        and _finite(frame["last_confirmed_swing_high"].iat[i])
        and float(frame["close"].iat[i]) > float(frame["last_confirmed_swing_high"].iat[i])
        and float(frame["close"].iat[i]) > sweep_extreme_opposite
    )


def annotate_post_news_external_liquidity_shift_frame(
    m3_frame: pd.DataFrame,
    m5_frame: pd.DataFrame,
    *,
    news_events: pd.DataFrame,
    news_config: NewsConfig,
    short_levels: tuple[tuple[str, str], ...] | None = None,
    long_levels: tuple[tuple[str, str], ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    result = m3_frame.copy()
    trade_window = _build_trade_window_contract(result.index, news_events, news_config)
    result = result.join(trade_window)
    result["els_signal"] = False
    result["els_direction"] = ""
    result["els_stop_price"] = np.nan
    result["els_source_kind"] = ""
    result["els_source_level_name"] = ""
    result["els_level_price"] = np.nan
    result["els_sweep_price"] = np.nan
    result["els_sweep_time_ny"] = ""
    result["els_confirm_time_ny"] = ""

    signal_rows: list[dict[str, Any]] = []
    for trade_day in sorted(pd.unique(result["trade_day"])):
        day_rows = result.loc[result["trade_day"] == trade_day]
        if day_rows.empty:
            continue
        tradable_from = int(day_rows["els_tradable_from_minute"].iloc[0])
        if bool(day_rows["els_day_consumed"].iloc[0]):
            continue

        m5_day = m5_frame.loc[m5_frame.index.date == trade_day].copy()
        m5_day = m5_day.loc[
            (m5_day.index.hour * 60 + m5_day.index.minute >= tradable_from)
            & (m5_day.index.hour * 60 + m5_day.index.minute < SWEEP_LATEST_MINUTE)
        ]
        if m5_day.empty:
            continue

        selected_sweep: dict[str, Any] | None = None
        ambiguous_day = False
        for sweep_ts, row in m5_day.iterrows():
            short_candidate = _directional_sweep_candidate(row, direction="short", short_levels=short_levels, long_levels=long_levels)
            long_candidate = _directional_sweep_candidate(row, direction="long", short_levels=short_levels, long_levels=long_levels)
            if short_candidate is not None and long_candidate is not None:

                ambiguous_day = True
                break
            candidate = short_candidate or long_candidate
            if candidate is None:
                continue
            level_name, source_kind, level_price, sweep_price, sweep_opposite_extreme = candidate
            selected_sweep = {
                "timestamp": sweep_ts,
                "direction": "short" if short_candidate is not None else "long",
                "level_name": level_name,
                "source_kind": source_kind,
                "level_price": level_price,
                "sweep_price": sweep_price,
                "sweep_opposite_extreme": sweep_opposite_extreme,
            }
            break
        if ambiguous_day or selected_sweep is None:
            continue

        confirm_candidates = day_rows.loc[day_rows.index > selected_sweep["timestamp"]].head(MAX_CONFIRM_BARS)
        if confirm_candidates.empty:
            continue

        invalidated = False
        for confirm_ts, _confirm_row in confirm_candidates.iterrows():
            intervening_m5 = m5_day.loc[(m5_day.index > selected_sweep["timestamp"]) & (m5_day.index <= confirm_ts)]
            if selected_sweep["direction"] == "short":
                if not intervening_m5.empty and bool((intervening_m5["close"] > selected_sweep["level_price"]).any()):
                    invalidated = True
                    break
            else:
                if not intervening_m5.empty and bool((intervening_m5["close"] < selected_sweep["level_price"]).any()):
                    invalidated = True
                    break

            i = result.index.get_loc(confirm_ts)
            if not _bar_confirmation(
                result,
                i,
                direction=selected_sweep["direction"],
                sweep_extreme_opposite=selected_sweep["sweep_opposite_extreme"],
            ):
                continue

            stop_price = (
                float(selected_sweep["sweep_price"]) + (STOP_BUFFER_PIPS * PIP_SIZE)
                if selected_sweep["direction"] == "short"
                else float(selected_sweep["sweep_price"]) - (STOP_BUFFER_PIPS * PIP_SIZE)
            )
            result.at[confirm_ts, "els_signal"] = True
            result.at[confirm_ts, "els_direction"] = selected_sweep["direction"]
            result.at[confirm_ts, "els_stop_price"] = stop_price
            result.at[confirm_ts, "els_source_kind"] = selected_sweep["source_kind"]
            result.at[confirm_ts, "els_source_level_name"] = selected_sweep["level_name"]
            result.at[confirm_ts, "els_level_price"] = float(selected_sweep["level_price"])
            result.at[confirm_ts, "els_sweep_price"] = float(selected_sweep["sweep_price"])
            result.at[confirm_ts, "els_sweep_time_ny"] = str(selected_sweep["timestamp"])
            result.at[confirm_ts, "els_confirm_time_ny"] = str(confirm_ts)
            signal_rows.append(
                {
                    "signal_time": confirm_ts,
                    "trade_day": trade_day,
                    "direction": selected_sweep["direction"],
                    "source_kind": selected_sweep["source_kind"],
                    "source_level_name": selected_sweep["level_name"],
                    "level_price": float(selected_sweep["level_price"]),
                    "sweep_price": float(selected_sweep["sweep_price"]),
                    "sweep_time_ny": str(selected_sweep["timestamp"]),
                    "confirm_time_ny": str(confirm_ts),
                    "tradable_from_hhmm": day_rows["els_tradable_from_hhmm"].iloc[0],
                }
            )
            break

        if invalidated:
            continue

    signal_log = pd.DataFrame(signal_rows)
    return result, signal_log


def build_research_frame(
    start: str,
    end: str,
    *,
    news_events: pd.DataFrame,
    news_config: NewsConfig,
    short_levels: tuple[tuple[str, str], ...] | None = None,
    long_levels: tuple[tuple[str, str], ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    filtered = _filtered_high_precision_package(start, end)
    m3_frame, m5_frame = _build_m3_m5_frames(filtered["mid_m1"])
    common_index = m3_frame.index.intersection(resample_ohlcv_to_timeframe(filtered["bid_m1"], TIMEFRAME).index).intersection(
        resample_ohlcv_to_timeframe(filtered["ask_m1"], TIMEFRAME).index
    )
    annotated, signal_log = annotate_post_news_external_liquidity_shift_frame(
        m3_frame.loc[common_index].copy(),
        m5_frame.copy(),
        news_events=news_events,
        news_config=news_config,
        short_levels=short_levels,
        long_levels=long_levels,
    )

    precision_package = _align_precision_package(filtered, annotated.index)
    return annotated, precision_package, signal_log


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=3)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def _filter_signal_log(signal_log: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if signal_log.empty:
        return signal_log.copy()
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    result = signal_log.copy()
    result["signal_time"] = pd.to_datetime(result["signal_time"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    return result.loc[(result["signal_time"] >= start_ts) & (result["signal_time"] <= end_ts)].copy()


def _merge_trade_sources(trades_export: pd.DataFrame, signal_log: pd.DataFrame) -> pd.DataFrame:
    if trades_export.empty:
        return trades_export.copy()
    trades = trades_export.copy()
    # Ensure consistent parsing and resolution (millisecond)
    trades["signal_time"] = pd.to_datetime(trades["signal_time_ny"], errors="coerce").dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward").dt.floor("ms")
    
    if signal_log.empty:
        trades["source_kind"] = ""
        trades["source_level_name"] = ""
        return trades
        
    s_log = signal_log.copy()
    s_log["signal_time"] = pd.to_datetime(s_log["signal_time"]).dt.floor("ms")
    
    merged = trades.merge(
        s_log[["signal_time", "source_kind", "source_level_name"]],
        on="signal_time",
        how="left",
    )
    merged["source_kind"] = merged["source_kind"].fillna("")
    merged["source_level_name"] = merged["source_level_name"].fillna("")
    return merged



def _source_split(trades_export: pd.DataFrame) -> pd.DataFrame:
    if trades_export.empty or "source_kind" not in trades_export.columns:
        return pd.DataFrame(columns=["source_kind", "trades", "win_rate", "profit_factor", "expectancy_r", "total_pnl_r"])
    rows: list[dict[str, Any]] = []
    for source_kind, chunk in trades_export.groupby("source_kind", dropna=False):
        if str(source_kind or "").strip() == "":
            continue
        pnl_r = pd.to_numeric(chunk["pnl_r"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "source_kind": str(source_kind),
                "trades": int(len(chunk)),
                "win_rate": float((pnl_r > 0).mean() * 100) if len(chunk) else 0.0,
                "profit_factor": _profit_factor(pnl_r),
                "expectancy_r": float(pnl_r.mean()) if len(chunk) else 0.0,
                "total_pnl_r": float(pnl_r.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("source_kind").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["source_kind", "trades", "win_rate", "profit_factor", "expectancy_r", "total_pnl_r"]
    )


def _side_split(trades_export: pd.DataFrame) -> pd.DataFrame:
    if trades_export.empty:
        return pd.DataFrame(columns=["direction", "trades", "win_rate", "profit_factor", "expectancy_r", "total_pnl_r"])
    rows: list[dict[str, Any]] = []
    for direction, chunk in trades_export.groupby("direction", dropna=False):
        pnl_r = pd.to_numeric(chunk["pnl_r"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "direction": str(direction),
                "trades": int(len(chunk)),
                "win_rate": float((pnl_r > 0).mean() * 100) if len(chunk) else 0.0,
                "profit_factor": _profit_factor(pnl_r),
                "expectancy_r": float(pnl_r.mean()) if len(chunk) else 0.0,
                "total_pnl_r": float(pnl_r.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("direction").reset_index(drop=True)


def news_metrics_from_summary(summary: dict[str, Any], trades_export: pd.DataFrame, signal_log: pd.DataFrame, news_block: np.ndarray, period_frame: pd.DataFrame) -> dict[str, Any]:
    news_exit_count = int((trades_export["exit_reason"] == "news_fortress_kill").sum()) if not trades_export.empty else 0
    blocked_signal_count = 0
    if not signal_log.empty:
        indexer = period_frame.index.get_indexer(signal_log["signal_time"])
        blocked = []
        for idx in indexer:
            fill_idx = idx + 1
            blocked.append(bool(fill_idx < len(news_block) and news_block[fill_idx]))
        signal_log["blocked_fill_by_news"] = blocked
        blocked_signal_count = int(sum(blocked))
    return {
        **summary,
        "news_exit_count": news_exit_count,
        "blocked_signal_count": blocked_signal_count,
    }


def evaluate_period(
    *,
    frame: pd.DataFrame,
    precision_package: dict[str, pd.DataFrame],
    signal_log: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: NewsConfig,
    start: str,
    end: str,
) -> dict[str, Any]:
    period_frame = period_slice(frame, start, end)
    if period_frame.empty:
        raise ValueError(f"Periodo vacio para {start} -> {end}")
    period_precision = _align_precision_package(
        {"bid_m1": precision_package["bid_m1"], "ask_m1": precision_package["ask_m1"], "mid_m1": precision_package["mid_m1"]},
        period_frame.index,
    )
    period_signal_log = _filter_signal_log(signal_log, start, end)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        strategy_module=strategy_module,
        frame=period_frame,
        params=params,
        engine_config=engine_config,
        news_block=news_block,
        news_filter_used=news_result.enabled,
        precision_package=period_precision,
        data_source_used="dukascopy_m1_bid_ask_full",
        news_events=news_result.events,
        news_settings=news_config,
    )
    summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
        strategy_module.NAME,
        result.trades,
        result.equity_curve,
        params,
        news_result.enabled,
        INITIAL_CAPITAL,
        None,
        costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
        timeframe=TIMEFRAME,
        schedule_used=schedule_used(),
        break_even_setting=params.get("break_even_at_r"),
    )
    trades_export = _merge_trade_sources(trades_export, period_signal_log)
    summary = news_metrics_from_summary(summary, trades_export, period_signal_log.copy(), news_block, period_frame)
    return {
        "summary": summary,
        "trades_export": trades_export,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
        "signal_log": period_signal_log,
        "source_split": _source_split(trades_export),
        "side_split": _side_split(trades_export),
    }


def selection_score(summary: dict[str, Any]) -> float:
    profit_factor = float(summary["profit_factor"])
    pf_for_score = 3.0 if not math.isfinite(profit_factor) else min(profit_factor, 3.0)
    score = 0.0
    score += pf_for_score * 150.0
    score += float(summary["expectancy_r"]) * 1200.0
    score += float(summary["total_return_pct"]) * 1.0
    score -= float(summary["max_drawdown_pct"]) * 3.0
    score -= float(summary["negative_years"]) * 35.0
    score -= float(summary["blocked_signal_count"]) * 8.0
    total_trades = int(summary["total_trades"])
    if total_trades < 12:
        score -= 800.0
    elif total_trades < SERIOUS_DEV_MIN_TRADES:
        score -= 180.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 100.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 100.0
    return score


def serious_gate_from_periods(dev: dict[str, Any], val: dict[str, Any], hold: dict[str, Any]) -> bool:
    return (
        int(dev["total_trades"]) >= SERIOUS_DEV_MIN_TRADES
        and float(dev["profit_factor"]) >= SERIOUS_DEV_MIN_PF
        and float(dev["expectancy_r"]) >= SERIOUS_DEV_MIN_EXPECTANCY
        and int(val["total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and int(hold["total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and float(val["profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(hold["profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(val["expectancy_r"]) >= SERIOUS_MIN_VAL_HOLD_EXPECTANCY
        and float(hold["expectancy_r"]) >= SERIOUS_MIN_VAL_HOLD_EXPECTANCY
    )


def decision_verdict(dev: dict[str, Any], val: dict[str, Any], hold: dict[str, Any], full: dict[str, Any]) -> str:
    if serious_gate_from_periods(dev, val, hold):
        return "structurally-promising-but-not-promotable"
    if int(full["total_trades"]) < SERIOUS_MIN_VAL_HOLD_TRADES or (
        float(full["profit_factor"]) > 1.0 and float(full["expectancy_r"]) > 0.0
    ):
        return "research-continues"
    return "hypothesis-falsified"


def main() -> Path:
    am_summary = build_am_grade_news_dataset()
    if am_summary["module_verdict"] != "READY_FOR_STRICT_AM_RESEARCH":
        raise RuntimeError(
            "La hipotesis eurusd_am_post_news_external_liquidity_shift requiere compuerta AM aprobada. "
            f"Veredicto actual={am_summary['module_verdict']}."
        )

    output_root = build_output_root()
    engine_config = build_engine_config()
    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context=strategy_module.NAME)
    full_frame, full_precision_package, full_signal_log = build_research_frame(
        *PERIODS["full_2020_2025"],
        news_events=news_result.events,
        news_config=news_config,
    )

    params = strategy_module.default_params()
    period_results: dict[str, dict[str, Any]] = {}
    for label, (start, end) in PERIODS.items():
        period_results[label] = evaluate_period(
            frame=full_frame,
            precision_package=full_precision_package,
            signal_log=full_signal_log,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=start,
            end=end,
        )

    dev = period_results["development_2020_2023"]["summary"]
    val = period_results["validation_2024"]["summary"]
    hold = period_results["holdout_2025"]["summary"]
    full = period_results["full_2020_2025"]["summary"]
    verdict = decision_verdict(dev, val, hold, full)

    scorecard = pd.DataFrame(
        [
            {
                "variant_label": params["variant_label"],
                "selection_score_dev": selection_score(dev),
                "dev_total_trades": dev["total_trades"],
                "dev_profit_factor": dev["profit_factor"],
                "dev_expectancy_r": dev["expectancy_r"],
                "val_total_trades": val["total_trades"],
                "val_profit_factor": val["profit_factor"],
                "val_expectancy_r": val["expectancy_r"],
                "hold_total_trades": hold["total_trades"],
                "hold_profit_factor": hold["profit_factor"],
                "hold_expectancy_r": hold["expectancy_r"],
                "full_total_trades": full["total_trades"],
                "full_profit_factor": full["profit_factor"],
                "full_expectancy_r": full["expectancy_r"],
                "full_max_drawdown_pct": full["max_drawdown_pct"],
                "verdict": verdict,
                "parameter_set_used": json.dumps(params, ensure_ascii=False, sort_keys=True),
            }
        ]
    )

    strategy_dir = output_root / strategy_module.NAME
    export_strategy_bundle(
        strategy_dir,
        summary=period_results["full_2020_2025"]["summary"],
        trades_export=period_results["full_2020_2025"]["trades_export"],
        monthly_stats=period_results["full_2020_2025"]["monthly_stats"],
        yearly_stats=period_results["full_2020_2025"]["yearly_stats"],
        equity_export=period_results["full_2020_2025"]["equity_export"],
        optimization_results=scorecard,
        extra_frames={
            "signal_log.csv": period_results["full_2020_2025"]["signal_log"],
            "source_split.csv": period_results["full_2020_2025"]["source_split"],
            "side_split.csv": period_results["full_2020_2025"]["side_split"],
        },
        extra_json={
            "selected_params.json": params,
            "period_summaries.json": {label: payload["summary"] for label, payload in period_results.items()},
            "serious_gate.json": {
                "development": {
                    "min_trades": SERIOUS_DEV_MIN_TRADES,
                    "min_profit_factor": SERIOUS_DEV_MIN_PF,
                    "min_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                },
                "validation_holdout": {
                    "min_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                    "min_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                    "min_expectancy_r": SERIOUS_MIN_VAL_HOLD_EXPECTANCY,
                },
            },
            "frame_contract.json": {
                "strategy": strategy_module.NAME,
                "pair": PAIR,
                "timeframe": TIMEFRAME,
                "sweep_frame": CONTEXT_TIMEFRAME,
                "operating_window": "07:00-11:00",
                "tradable_from_cutoff": "10:45",
                "asia_window": "19:00-03:00",
                "london_window": "03:00-07:00",
                "min_sweep_pips": MIN_SWEEP_PIPS,
                "max_sweep_atr_mult": MAX_SWEEP_ATR_MULT,
                "max_confirm_bars": MAX_CONFIRM_BARS,
                "stop_buffer_pips": STOP_BUFFER_PIPS,
                "target_rr": TARGET_RR,
                "break_even_at_r": BREAK_EVEN_AT_R,
                "max_hold_bars": MAX_HOLD_BARS,
            },
            "verdict.json": {"verdict": verdict},
        },
    )

    scorecard.to_csv(output_root / f"{strategy_module.NAME}_scorecard.csv", index=False)
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "strategy": strategy_module.NAME,
                "pair": PAIR,
                "timeframe": TIMEFRAME,
                "periods": PERIODS,
                "schedule_used": schedule_used(),
                "engine_config": {
                    "execution_mode": engine_config.execution_mode,
                    "cost_profile": engine_config.cost_profile,
                    "session_cutoff": engine_config.session_cutoff,
                },
                "news_dataset": str(news_result.final_dataset_path),
                "news_enabled": news_result.enabled,
                "verdict": verdict,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_root


if __name__ == "__main__":
    root = main()
    print(f"Backtest canonico completado en: {root}")
