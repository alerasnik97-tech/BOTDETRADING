from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DATA_DIRS,
    DEFAULT_NEWS_FILE,
    DEFAULT_PAIR,
    DEFAULT_RESULTS_DIR,
    INITIAL_CAPITAL,
    NY_TZ,
    PAIR_META,
    SLIPPAGE_PIPS,
    NewsConfig,
    SessionConfig,
    StrategyParams,
    optimization_grid,
)
from news_filter import build_entry_block, load_news_events
from report import (
    build_equity_curve_export,
    build_hourly_stats,
    build_period_stats,
    build_summary,
    build_trades_export,
    export_chatgpt_bundle,
    print_console_report,
)


PREVIOUS_RESULTS_DIR = Path("results_simple_session_hybrid_main")


@dataclass
class Position:
    direction: str
    module: str
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    units: float
    risk_usd: float


@dataclass
class BacktestRun:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict[str, Any]
    yearly_stats: pd.DataFrame
    news_filter_used: bool
    regime_counts: dict[str, int]
    selected_score: float | None = None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def bollinger_bands(series: pd.Series, period: int, std_mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + std * std_mult
    lower = mid - std * std_mult
    return mid, upper, lower


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(frame: pd.DataFrame, period: int) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    atr_series = atr(frame, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series.replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def load_price_data(pair: str, data_dirs: list[Path], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for data_dir in data_dirs:
        path = data_dir / f"{pair}_M5.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        idx = pd.to_datetime(frame.index, utc=True).tz_convert(NY_TZ)
        frame.index = idx
        frames.append(frame[["open", "high", "low", "close", "volume"]].copy())

    if not frames:
        raise FileNotFoundError(f"No encontré datos preparados para {pair} en: {data_dirs}")

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[merged.index.dayofweek < 5].copy()
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    return merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()


def build_h1_context(frame: pd.DataFrame, slope_lookback: int) -> pd.DataFrame:
    h1 = (
        frame.resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    h1["atr14"] = atr(h1, 14)
    h1["adx14"] = adx(h1, 14)
    h1["ema50"] = ema(h1["close"], 50)
    h1["ema200"] = ema(h1["close"], 200)
    h1["ema200_slope"] = h1["ema200"] - h1["ema200"].shift(slope_lookback)
    h1["trend_progress_atr"] = (h1["close"] - h1["close"].shift(slope_lookback)) / h1["atr14"].replace(0.0, np.nan)
    aligned = h1[["atr14", "adx14", "ema50", "ema200", "ema200_slope", "trend_progress_atr"]].reindex(frame.index, method="ffill")
    return aligned.rename(
        columns={
            "atr14": "h1_atr14",
            "adx14": "h1_adx14",
            "ema50": "h1_ema50",
            "ema200": "h1_ema200",
            "ema200_slope": "h1_ema200_slope",
            "trend_progress_atr": "h1_trend_progress_atr",
        }
    )


def build_feature_frame(frame: pd.DataFrame, slope_lookback: int, compression_bars: int) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["ema20"] = ema(enriched["close"], 20)
    enriched["atr14"] = atr(enriched, 14)
    enriched["rsi9"] = rsi(enriched["close"], 9)
    enriched["rsi14"] = rsi(enriched["close"], 14)
    bb_mid, bb_upper, bb_lower = bollinger_bands(enriched["close"], 20, 2.0)
    enriched["bb_mid_20_2_0"] = bb_mid
    enriched["bb_upper_20_2_0"] = bb_upper
    enriched["bb_lower_20_2_0"] = bb_lower
    bb_mid_22, bb_upper_22, bb_lower_22 = bollinger_bands(enriched["close"], 20, 2.2)
    enriched["bb_mid_20_2_2"] = bb_mid_22
    enriched["bb_upper_20_2_2"] = bb_upper_22
    enriched["bb_lower_20_2_2"] = bb_lower_22
    enriched["bar_range"] = enriched["high"] - enriched["low"]
    enriched["range_atr"] = enriched["bar_range"] / enriched["atr14"].replace(0.0, np.nan)
    enriched["prev_close"] = enriched["close"].shift(1)
    enriched["compression_high"] = enriched["high"].shift(1).rolling(compression_bars).max()
    enriched["compression_low"] = enriched["low"].shift(1).rolling(compression_bars).min()
    enriched["compression_range"] = enriched["compression_high"] - enriched["compression_low"]
    enriched["compression_range_atr"] = enriched["compression_range"] / enriched["atr14"].shift(1).replace(0.0, np.nan)
    context = build_h1_context(enriched, slope_lookback)
    enriched = enriched.join(context)
    return enriched.dropna().copy()


def time_to_minute(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def quote_to_usd(pair: str, pair_price: float) -> float:
    quote = PAIR_META[pair]["quote"]
    if quote == "USD":
        return 1.0
    if pair == "USDJPY":
        return 1.0 / pair_price if pair_price > 0 else np.nan
    raise ValueError(f"No hay conversión a USD implementada para {pair}")


def estimate_spread_pips(pair: str, range_atr: float) -> float:
    base_spread = float(PAIR_META[pair]["default_spread_pips"])
    multiplier = float(np.clip(range_atr, 0.8, 2.5)) if np.isfinite(range_atr) else 1.0
    return base_spread * multiplier


def execution_adjustment(pair: str, spread_pips: float) -> float:
    pip_size = PAIR_META[pair]["pip_size"]
    return (spread_pips * 0.5 + SLIPPAGE_PIPS) * pip_size


def score_result(summary: dict[str, Any]) -> float:
    total_trades = float(summary["total_trades"])
    trades_per_month = float(summary["avg_trades_per_month"])
    profit_factor_raw = float(summary["profit_factor"])
    profit_factor_capped = 2.5 if not np.isfinite(profit_factor_raw) else min(profit_factor_raw, 2.5)
    expectancy_r = float(summary["expectancy_r"])
    score = (
        profit_factor_capped * 220.0
        + expectancy_r * 650.0
        + float(summary["total_return_pct"]) * 1.0
        - float(summary["max_drawdown_pct"]) * 5.0
        - float(summary["negative_months"]) * 2.5
        - float(summary["negative_years"]) * 22.0
    )
    if profit_factor_raw < 1.0:
        score -= 110.0
    if expectancy_r < 0.0:
        score -= 110.0
    if total_trades < 50.0:
        score -= 2000.0
    if trades_per_month < 4.0:
        score -= (4.0 - trades_per_month) * 80.0
    if total_trades < 100.0:
        score -= (100.0 - total_trades) * 3.0
    if total_trades < 12.0:
        score -= 500.0
    return score


def build_output_dir(results_dir: Path, pair: str, mode: str) -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir / f"{timestamp}_{pair.lower()}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_optimization_row(summary: dict[str, Any], params: StrategyParams, score: float) -> dict[str, Any]:
    row = params.to_dict()
    row.update(summary)
    row["score"] = score
    return row


def load_previous_summary() -> dict[str, Any] | None:
    if not PREVIOUS_RESULTS_DIR.exists():
        return None
    candidates = sorted(PREVIOUS_RESULTS_DIR.glob("*/PARA CHATGPT/summary.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def build_run_summary(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    params: StrategyParams,
    news_filter_used: bool,
    regime_counts: dict[str, int],
    selected_score: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trades_export = build_trades_export(trades_df)
    yearly_stats = build_period_stats(trades_export, "Y", INITIAL_CAPITAL)
    monthly_stats = build_period_stats(trades_export, "M", INITIAL_CAPITAL)
    hourly_stats = build_hourly_stats(trades_export)
    equity_export = build_equity_curve_export(equity_df)
    summary = build_summary(
        trades_export,
        trades_df,
        equity_export,
        monthly_stats,
        yearly_stats,
        hourly_stats,
        parameter_set_used=params.to_dict(),
        news_filter_used=news_filter_used,
        regime_counts=regime_counts,
        selected_score=selected_score,
    )
    return summary, yearly_stats


def classify_regimes(frame: pd.DataFrame, params: StrategyParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h1_ema50 = frame["h1_ema50"].to_numpy()
    h1_ema200 = frame["h1_ema200"].to_numpy()
    h1_adx14 = frame["h1_adx14"].to_numpy()
    h1_atr14 = frame["h1_atr14"].to_numpy()
    h1_ema200_slope = frame["h1_ema200_slope"].to_numpy()
    h1_trend_progress_atr = frame["h1_trend_progress_atr"].to_numpy()
    ema_distance_atr = np.abs(h1_ema50 - h1_ema200) / np.where(h1_atr14 > 0, h1_atr14, np.nan)
    trend_base = (
        np.isfinite(h1_ema50)
        & np.isfinite(h1_ema200)
        & np.isfinite(h1_adx14)
        & np.isfinite(h1_ema200_slope)
        & np.isfinite(ema_distance_atr)
        & np.isfinite(h1_trend_progress_atr)
    )
    trend_strength = trend_base & (h1_adx14 >= params.adx_trend_min) & (ema_distance_atr >= params.ema_distance_atr_min)
    trend_up = trend_strength & (h1_ema50 > h1_ema200) & (h1_ema200_slope > 0) & (h1_trend_progress_atr >= params.trend_progress_atr_min)
    trend_down = trend_strength & (h1_ema50 < h1_ema200) & (h1_ema200_slope < 0) & (h1_trend_progress_atr <= -params.trend_progress_atr_min)
    range_weak = ~(trend_up | trend_down)
    return trend_up, trend_down, range_weak


def run_backtest(frame: pd.DataFrame, pair: str, params: StrategyParams, news_block: np.ndarray, news_filter_used: bool) -> BacktestRun:
    session = SessionConfig()
    force_close_minute = time_to_minute(session.force_close)
    entry_start_minute = time_to_minute(session.entry_start)
    entry_end_minute = time_to_minute(session.entry_end)
    entry_cutoff_minute = min(entry_end_minute, force_close_minute - session.entry_cutoff_minutes)

    local_index = frame.index.tz_convert(NY_TZ)
    timestamp_utc = frame.index.tz_convert("UTC")
    minute_values = (local_index.hour * 60 + local_index.minute).to_numpy()
    session_dates = np.array(local_index.date)

    open_ = frame["open"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    prev_close = frame["prev_close"].to_numpy()
    ema20 = frame["ema20"].to_numpy()
    atr14 = frame["atr14"].to_numpy()
    range_atr = frame["range_atr"].to_numpy()
    compression_high = frame["compression_high"].to_numpy()
    compression_low = frame["compression_low"].to_numpy()
    compression_range_atr = frame["compression_range_atr"].to_numpy()

    bb_suffix = "20_2_2" if abs(params.bb_std - 2.2) < 1e-9 else "20_2_0"
    bb_upper = frame[f"bb_upper_{bb_suffix}"].to_numpy()
    bb_lower = frame[f"bb_lower_{bb_suffix}"].to_numpy()
    bb_mid = frame[f"bb_mid_{bb_suffix}"].to_numpy()
    bb_width_atr = (bb_upper - bb_lower) / np.where(atr14 > 0, atr14, np.nan)
    rsi_col = "rsi14" if params.range_rsi_period == 14 else "rsi9"
    range_rsi = frame[rsi_col].to_numpy()

    trend_up, trend_down, range_weak = classify_regimes(frame, params)
    regime_counts = {
        "trend_up": int(trend_up.sum()),
        "trend_down": int(trend_down.sum()),
        "range_weak": int(range_weak.sum()),
    }

    breakout_enabled = params.model_mode in {"hybrid", "breakout_only"} and params.breakout_enabled
    range_enabled = params.model_mode in {"hybrid", "range_only"}

    entry_allowed = (minute_values >= entry_start_minute) & (minute_values <= entry_cutoff_minute)
    force_close_mask = minute_values >= force_close_minute

    cash = INITIAL_CAPITAL
    position: Position | None = None
    pending_direction: str | None = None
    pending_module: str | None = None
    pending_signal_index: int | None = None
    trades: list[dict[str, Any]] = []
    equity_points: list[dict[str, Any]] = [{"timestamp": timestamp_utc[0], "equity": INITIAL_CAPITAL}]
    opened_total_by_date: dict[Any, int] = {}
    opened_direction_by_date: dict[Any, dict[str, int]] = {}
    daily_realized_r: dict[Any, float] = {}
    halted_dates: set[Any] = set()
    cooldown_until_index = -1

    for i in range(3, len(frame) - 1):
        ts_utc = timestamp_utc[i]
        session_date = session_dates[i]

        if pending_direction is not None and pending_signal_index is not None and pending_module is not None and i == pending_signal_index + 1:
            signal_range_atr = range_atr[pending_signal_index]
            spread_pips = estimate_spread_pips(pair, signal_range_atr)
            if (
                entry_allowed[i]
                and not news_block[i]
                and session_date not in halted_dates
                and np.isfinite(atr14[pending_signal_index])
                and atr14[pending_signal_index] > 0
                and signal_range_atr <= params.shock_candle_atr_max
                and spread_pips <= params.max_spread_pips
            ):
                entry_adjustment = execution_adjustment(pair, spread_pips)
                entry_price = open_[i] + entry_adjustment if pending_direction == "long" else open_[i] - entry_adjustment
                if pending_module == "breakout":
                    if pending_direction == "long":
                        if params.breakout_stop_mode == "compression_stop":
                            sl = compression_low[pending_signal_index] - atr14[pending_signal_index] * params.breakout_stop_atr
                            stop_distance = entry_price - sl
                        else:
                            stop_distance = atr14[pending_signal_index] * params.breakout_stop_atr
                            sl = entry_price - stop_distance
                    else:
                        if params.breakout_stop_mode == "compression_stop":
                            sl = compression_high[pending_signal_index] + atr14[pending_signal_index] * params.breakout_stop_atr
                            stop_distance = sl - entry_price
                        else:
                            stop_distance = atr14[pending_signal_index] * params.breakout_stop_atr
                            sl = entry_price + stop_distance
                    target_rr = params.breakout_target_rr
                else:
                    stop_distance = atr14[pending_signal_index] * params.range_stop_atr
                    sl = entry_price - stop_distance if pending_direction == "long" else entry_price + stop_distance
                    target_rr = params.range_target_rr

                quote_to_usd_rate = quote_to_usd(pair, entry_price)
                if np.isfinite(stop_distance) and stop_distance > 0 and np.isfinite(quote_to_usd_rate) and quote_to_usd_rate > 0:
                    risk_usd = INITIAL_CAPITAL * (params.risk_pct / 100.0)
                    units = math.floor(risk_usd / (stop_distance * quote_to_usd_rate))
                    if units > 0:
                        tp = entry_price + stop_distance * target_rr if pending_direction == "long" else entry_price - stop_distance * target_rr
                        position = Position(
                            direction=pending_direction,
                            module=pending_module,
                            entry_time=ts_utc,
                            entry_price=entry_price,
                            sl=sl,
                            tp=tp,
                            units=float(units),
                            risk_usd=risk_usd,
                        )
                        opened_total_by_date[session_date] = opened_total_by_date.get(session_date, 0) + 1
                        opened_direction_by_date.setdefault(session_date, {"long": 0, "short": 0})
                        opened_direction_by_date[session_date][pending_direction] += 1
            pending_direction = None
            pending_module = None
            pending_signal_index = None

        if position is not None:
            bar_spread_pips = estimate_spread_pips(pair, range_atr[i])
            exit_adjustment = execution_adjustment(pair, bar_spread_pips)
            stop_distance = abs(position.entry_price - position.sl)
            if position.module == "breakout" and params.breakout_be_enabled:
                if position.direction == "long" and high[i] >= position.entry_price + stop_distance:
                    position.sl = max(position.sl, position.entry_price)
                elif position.direction == "short" and low[i] <= position.entry_price - stop_distance:
                    position.sl = min(position.sl, position.entry_price)
            if position.module == "range_mr" and params.range_be_enabled:
                if position.direction == "long" and high[i] >= position.entry_price + stop_distance:
                    position.sl = max(position.sl, position.entry_price)
                elif position.direction == "short" and low[i] <= position.entry_price - stop_distance:
                    position.sl = min(position.sl, position.entry_price)

            exit_reason = None
            exit_price = None
            unrealized_r = 0.0
            if position.direction == "long":
                unrealized_r = ((close[i] - position.entry_price) / stop_distance) if stop_distance > 0 else 0.0
            else:
                unrealized_r = ((position.entry_price - close[i]) / stop_distance) if stop_distance > 0 else 0.0
            if daily_realized_r.get(session_date, 0.0) + unrealized_r <= -params.daily_loss_limit_r:
                exit_reason = "daily_circuit_breaker"
                exit_price = close[i] - exit_adjustment if position.direction == "long" else close[i] + exit_adjustment
                halted_dates.add(session_date)
            elif force_close_mask[i]:
                exit_reason = "forced_session_close"
                exit_price = close[i] - exit_adjustment if position.direction == "long" else close[i] + exit_adjustment
            elif position.direction == "long":
                if low[i] <= position.sl:
                    exit_reason = "stop_loss"
                    exit_price = position.sl
                elif high[i] >= position.tp:
                    exit_reason = "take_profit"
                    exit_price = position.tp
            else:
                if high[i] >= position.sl:
                    exit_reason = "stop_loss"
                    exit_price = position.sl
                elif low[i] <= position.tp:
                    exit_reason = "take_profit"
                    exit_price = position.tp

            if exit_reason is not None and exit_price is not None:
                price_delta = exit_price - position.entry_price
                pnl_quote = price_delta * position.units if position.direction == "long" else -price_delta * position.units
                pnl_usd = pnl_quote * quote_to_usd(pair, exit_price)
                pnl_r = pnl_usd / position.risk_usd if position.risk_usd else 0.0
                cash += pnl_usd
                daily_realized_r[session_date] = daily_realized_r.get(session_date, 0.0) + pnl_r
                if daily_realized_r[session_date] <= -params.daily_loss_limit_r:
                    halted_dates.add(session_date)
                trades.append(
                    {
                        "pair": pair,
                        "module": position.module,
                        "entry_time": position.entry_time,
                        "exit_time": ts_utc,
                        "direction": position.direction,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "sl": position.sl,
                        "tp": position.tp,
                        "pnl_r": pnl_r,
                        "pnl_usd": pnl_usd,
                        "exit_reason": exit_reason,
                    }
                )
                cooldown_until_index = i + params.cooldown_bars
                position = None

        if position is None and pending_direction is None:
            if i <= cooldown_until_index or not entry_allowed[i] or news_block[i] or session_date in halted_dates:
                pass
            elif opened_total_by_date.get(session_date, 0) >= params.max_trades_per_day:
                pass
            else:
                opened_direction_by_date.setdefault(session_date, {"long": 0, "short": 0})
                shock_ok = np.isfinite(range_atr[i]) and range_atr[i] <= params.shock_candle_atr_max
                spread_pips = estimate_spread_pips(pair, range_atr[i])
                spread_ok = spread_pips <= params.max_spread_pips

                breakout_signal_range_ok = np.isfinite(range_atr[i]) and range_atr[i] <= params.breakout_candle_atr_max
                compression_ok = np.isfinite(compression_range_atr[i]) and compression_range_atr[i] <= params.compression_atr_mult
                comp_mid_distance_atr = abs(((compression_high[i] + compression_low[i]) * 0.5) - ema20[i]) / atr14[i] if atr14[i] > 0 else np.nan
                recent_up_bars = int(np.sum(close[i - 3 : i] > prev_close[i - 3 : i]))
                recent_down_bars = int(np.sum(close[i - 3 : i] < prev_close[i - 3 : i]))
                body_ratio = abs(close[i] - open_[i]) / max(high[i] - low[i], 1e-9)

                long_breakout = (
                    breakout_enabled
                    and trend_up[i]
                    and compression_ok
                    and breakout_signal_range_ok
                    and shock_ok
                    and spread_ok
                    and close[i] > compression_high[i]
                    and close[i - 1] <= compression_high[i]
                    and (close[i] >= ema20[i] if params.require_breakout_above_below_ema20 else True)
                    and np.isfinite(comp_mid_distance_atr)
                    and comp_mid_distance_atr <= 0.8
                    and recent_up_bars <= 2
                    and body_ratio >= 0.4
                    and opened_direction_by_date[session_date]["long"] < params.max_trades_per_direction
                )
                short_breakout = (
                    breakout_enabled
                    and trend_down[i]
                    and compression_ok
                    and breakout_signal_range_ok
                    and shock_ok
                    and spread_ok
                    and close[i] < compression_low[i]
                    and close[i - 1] >= compression_low[i]
                    and (close[i] <= ema20[i] if params.require_breakout_above_below_ema20 else True)
                    and np.isfinite(comp_mid_distance_atr)
                    and comp_mid_distance_atr <= 0.8
                    and recent_down_bars <= 2
                    and body_ratio >= 0.4
                    and opened_direction_by_date[session_date]["short"] < params.max_trades_per_direction
                )

                band_width_ok = np.isfinite(bb_width_atr[i]) and (bb_width_atr[i] >= 1.0) and (bb_width_atr[i] <= 4.0)
                long_range = (
                    range_enabled
                    and range_weak[i]
                    and shock_ok
                    and spread_ok
                    and np.isfinite(range_atr[i])
                    and range_atr[i] <= params.range_signal_candle_atr_max
                    and band_width_ok
                    and low[i] <= bb_lower[i]
                    and close[i] >= bb_lower[i]
                    and range_rsi[i] <= params.range_rsi_low
                    and close[i] > prev_close[i]
                    and opened_direction_by_date[session_date]["long"] < params.max_trades_per_direction
                )
                short_range = (
                    range_enabled
                    and range_weak[i]
                    and shock_ok
                    and spread_ok
                    and np.isfinite(range_atr[i])
                    and range_atr[i] <= params.range_signal_candle_atr_max
                    and band_width_ok
                    and high[i] >= bb_upper[i]
                    and close[i] <= bb_upper[i]
                    and range_rsi[i] >= params.range_rsi_high
                    and close[i] < prev_close[i]
                    and opened_direction_by_date[session_date]["short"] < params.max_trades_per_direction
                )

                if long_breakout:
                    pending_direction = "long"
                    pending_module = "breakout"
                    pending_signal_index = i
                elif short_breakout:
                    pending_direction = "short"
                    pending_module = "breakout"
                    pending_signal_index = i
                elif long_range:
                    pending_direction = "long"
                    pending_module = "range_mr"
                    pending_signal_index = i
                elif short_range:
                    pending_direction = "short"
                    pending_module = "range_mr"
                    pending_signal_index = i

        mark_equity = cash
        if position is not None:
            unrealized_quote = (close[i] - position.entry_price) * position.units if position.direction == "long" else (position.entry_price - close[i]) * position.units
            unrealized_usd = unrealized_quote * quote_to_usd(pair, close[i])
            mark_equity += unrealized_usd
        equity_points.append({"timestamp": ts_utc, "equity": mark_equity})

    if position is not None:
        final_ts = timestamp_utc[-1]
        final_exit = close[-1]
        price_delta = final_exit - position.entry_price
        pnl_quote = price_delta * position.units if position.direction == "long" else -price_delta * position.units
        pnl_usd = pnl_quote * quote_to_usd(pair, final_exit)
        pnl_r = pnl_usd / position.risk_usd if position.risk_usd else 0.0
        cash += pnl_usd
        trades.append(
            {
                "pair": pair,
                "module": position.module,
                "entry_time": position.entry_time,
                "exit_time": final_ts,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": final_exit,
                "sl": position.sl,
                "tp": position.tp,
                "pnl_r": pnl_r,
                "pnl_usd": pnl_usd,
                "exit_reason": "final_bar_close",
            }
        )
        equity_points.append({"timestamp": final_ts, "equity": cash})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)
    summary, yearly_stats = build_run_summary(trades_df, equity_df, params, news_filter_used, regime_counts)
    return BacktestRun(
        trades=trades_df,
        equity_curve=equity_df,
        summary=summary,
        yearly_stats=yearly_stats,
        news_filter_used=news_filter_used,
        regime_counts=regime_counts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bot simple híbrido M5+H1 por régimen: breakout en tendencia y mean reversion en rango.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--pair", default=DEFAULT_PAIR)
        target.add_argument("--start", default="2020-01-01")
        target.add_argument("--end", default="2025-12-31")
        target.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
        target.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
        target.add_argument("--news-file", default=str(DEFAULT_NEWS_FILE))
        target.add_argument("--disable-news", action="store_true")
        target.add_argument("--news-pre-minutes", type=int, default=15)
        target.add_argument("--news-post-minutes", type=int, default=15)
        target.add_argument("--risk-pct", type=float, default=0.5)
        target.add_argument("--adx-trend-min", type=int, default=18)
        target.add_argument("--ema200-slope-lookback", type=int, default=5)
        target.add_argument("--trend-progress-atr-min", type=float, default=0.8)
        target.add_argument("--ema-distance-atr-min", type=float, default=0.10)
        target.add_argument("--model-mode", choices=["hybrid", "range_only", "breakout_only"], default="hybrid")
        target.add_argument("--bb-std", type=float, default=2.0)
        target.add_argument("--range-rsi-period", type=int, choices=[9, 14], default=9)
        target.add_argument("--range-signal-candle-atr-max", type=float, default=1.5)
        target.add_argument("--compression-bars", type=int, default=6)
        target.add_argument("--compression-atr-mult", type=float, default=1.0)
        target.add_argument("--breakout-candle-atr-max", type=float, default=1.5)
        target.add_argument("--breakout-stop-mode", choices=["atr_stop", "compression_stop"], default="atr_stop")
        target.add_argument("--breakout-stop-atr", type=float, default=1.0)
        target.add_argument("--breakout-target-rr", type=float, default=1.8)
        target.add_argument("--breakout-enabled", action="store_true")
        target.add_argument("--breakout-be-enabled", action="store_true")
        target.add_argument("--range-rsi-low", type=float, default=40.0)
        target.add_argument("--range-rsi-high", type=float, default=60.0)
        target.add_argument("--range-stop-atr", type=float, default=1.0)
        target.add_argument("--range-target-rr", type=float, default=1.3)
        target.add_argument("--range-be-enabled", action="store_true")
        target.add_argument("--cooldown-bars", type=int, default=3)
        target.add_argument("--daily-loss-limit-r", type=float, default=1.5)
        target.add_argument("--shock-candle-atr-max", type=float, default=2.2)
        target.add_argument("--max-spread-pips", type=float, default=1.2)

    run_parser = subparsers.add_parser("run", help="Corre un backtest simple.")
    add_common_arguments(run_parser)

    optimize_parser = subparsers.add_parser("optimize", help="Optimización compacta y rápida.")
    add_common_arguments(optimize_parser)
    optimize_parser.add_argument("--max-combinations", type=int, default=18)

    return parser


def build_params_from_args(args: argparse.Namespace) -> tuple[StrategyParams, NewsConfig]:
    params = StrategyParams(
        pair=args.pair.upper().strip(),
        risk_pct=args.risk_pct,
        model_mode=args.model_mode,
        adx_trend_min=args.adx_trend_min,
        ema200_slope_lookback=args.ema200_slope_lookback,
        trend_progress_atr_min=args.trend_progress_atr_min,
        ema_distance_atr_min=args.ema_distance_atr_min,
        bb_std=args.bb_std,
        range_rsi_period=args.range_rsi_period,
        range_signal_candle_atr_max=args.range_signal_candle_atr_max,
        compression_bars=args.compression_bars,
        compression_atr_mult=args.compression_atr_mult,
        breakout_candle_atr_max=args.breakout_candle_atr_max,
        breakout_stop_mode=args.breakout_stop_mode,
        breakout_stop_atr=args.breakout_stop_atr,
        breakout_target_rr=args.breakout_target_rr,
        range_rsi_low=args.range_rsi_low,
        range_rsi_high=args.range_rsi_high,
        range_stop_atr=args.range_stop_atr,
        range_target_rr=args.range_target_rr,
        range_be_enabled=args.range_be_enabled,
        breakout_be_enabled=args.breakout_be_enabled,
        breakout_enabled=args.breakout_enabled or args.model_mode != "range_only",
        cooldown_bars=args.cooldown_bars,
        daily_loss_limit_r=args.daily_loss_limit_r,
        shock_candle_atr_max=args.shock_candle_atr_max,
        max_spread_pips=args.max_spread_pips,
    )
    news_cfg = NewsConfig(
        enabled=not args.disable_news,
        file_path=Path(args.news_file),
        pre_minutes=args.news_pre_minutes,
        post_minutes=args.news_post_minutes,
    )
    return params, news_cfg


def run_single(args: argparse.Namespace) -> None:
    start_time = time.time()
    params, news_config = build_params_from_args(args)
    data_dirs = [Path(item) for item in args.data_dirs]
    raw_frame = load_price_data(params.pair, data_dirs, args.start, args.end)
    features = build_feature_frame(raw_frame, params.ema200_slope_lookback, params.compression_bars)
    news_events, news_filter_used = load_news_events(params.pair, news_config)
    news_block = build_entry_block(features.index, news_events, news_config)
    run = run_backtest(features, params.pair, params, news_block, news_filter_used)
    score = score_result(run.summary)
    run.summary["selected_score"] = score
    optimization_results = pd.DataFrame([make_optimization_row(run.summary, params, score)])
    output_dir = build_output_dir(Path(args.results_dir), params.pair, "hybrid_run")
    export_chatgpt_bundle(
        output_dir,
        trades=run.trades,
        equity_curve=run.equity_curve,
        params=params.to_dict(),
        news_filter_used=run.news_filter_used,
        optimization_results=optimization_results,
        initial_capital=INITIAL_CAPITAL,
        regime_counts=run.regime_counts,
        selected_score=score,
    )
    print_console_report(
        run.summary,
        run.yearly_stats,
        optimization_results,
        runtime_seconds=time.time() - start_time,
        previous_summary=load_previous_summary(),
    )


def run_optimize(args: argparse.Namespace) -> None:
    start_time = time.time()
    pair = args.pair.upper().strip()
    data_dirs = [Path(item) for item in args.data_dirs]
    raw_frame = load_price_data(pair, data_dirs, args.start, args.end)
    candidates = optimization_grid(pair, args.risk_pct, args.max_combinations)
    if not candidates:
        raise RuntimeError("La grilla de optimización no generó candidatos.")

    features_cache: dict[tuple[int, int], pd.DataFrame] = {}
    news_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, bool]] = {}
    rows: list[dict[str, Any]] = []
    best_run: BacktestRun | None = None
    best_params: StrategyParams | None = None
    best_score = -float("inf")

    for params, base_news_cfg in candidates:
        cache_key = (params.ema200_slope_lookback, params.compression_bars)
        if cache_key not in features_cache:
            features_cache[cache_key] = build_feature_frame(raw_frame, params.ema200_slope_lookback, params.compression_bars)
        features = features_cache[cache_key]
        news_key = (params.ema200_slope_lookback, params.compression_bars, base_news_cfg.pre_minutes, base_news_cfg.post_minutes)
        if news_key not in news_cache:
            effective_news_cfg = NewsConfig(
                enabled=not args.disable_news,
                file_path=Path(args.news_file),
                pre_minutes=base_news_cfg.pre_minutes,
                post_minutes=base_news_cfg.post_minutes,
            )
            news_events, news_filter_used = load_news_events(pair, effective_news_cfg)
            news_cache[news_key] = (build_entry_block(features.index, news_events, effective_news_cfg), news_filter_used)
        news_block, news_filter_used = news_cache[news_key]
        run = run_backtest(features, pair, params, news_block, news_filter_used)
        score = score_result(run.summary)
        run.summary["selected_score"] = score
        rows.append(make_optimization_row(run.summary, params, score))
        if score > best_score:
            best_score = score
            best_run = run
            best_params = params

    if best_run is None or best_params is None:
        raise RuntimeError("No pude seleccionar una mejor combinación.")

    best_run.summary["selected_score"] = best_score
    optimization_results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    output_dir = build_output_dir(Path(args.results_dir), pair, "hybrid_optimize")
    export_chatgpt_bundle(
        output_dir,
        trades=best_run.trades,
        equity_curve=best_run.equity_curve,
        params=best_params.to_dict(),
        news_filter_used=best_run.news_filter_used,
        optimization_results=optimization_results,
        initial_capital=INITIAL_CAPITAL,
        regime_counts=best_run.regime_counts,
        selected_score=best_score,
    )
    print_console_report(
        best_run.summary,
        best_run.yearly_stats,
        optimization_results,
        runtime_seconds=time.time() - start_time,
        previous_summary=load_previous_summary(),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_single(args)
    elif args.command == "optimize":
        run_optimize(args)
    else:
        raise ValueError(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
