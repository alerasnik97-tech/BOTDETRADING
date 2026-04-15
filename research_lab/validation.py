from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import ALT_WFA_IS_MONTHS, ALT_WFA_OOS_MONTHS, DEFAULT_MAX_EVALS_PER_STRATEGY, DEFAULT_SEED, DEFAULT_WFA_IS_MONTHS, DEFAULT_WFA_OOS_MONTHS, INITIAL_CAPITAL, NewsConfig
from research_lab.data_loader import slice_high_precision_package_to_frame
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, load_news_events
from research_lab.report import summarize_result
from research_lab.scorer import score_is_summary


@dataclass
class WFAResult:
    fold_rows: pd.DataFrame
    oos_trades: pd.DataFrame
    oos_equity_curve: pd.DataFrame
    oos_summary: dict[str, Any]


def parameter_combinations(strategy_module: Any, max_evals: int = DEFAULT_MAX_EVALS_PER_STRATEGY, seed: int = DEFAULT_SEED) -> list[dict]:
    return strategy_module.parameter_grid(max_combinations=max_evals, seed=seed)


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz)


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    if pd.isna(ts):
        return pd.NaT
    naive = ts.tz_localize(None)
    period = naive.to_period("M") + months
    return period.to_timestamp().tz_localize(ts.tz)


def build_wfa_folds(frame: pd.DataFrame, is_months: int, oos_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if frame.empty:
        return []
    start_month = month_start(frame.index.min())
    last_month = month_start(frame.index.max())
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    train_start = start_month
    while True:
        train_end = add_months(train_start, is_months) - pd.Timedelta(minutes=15)
        test_start = add_months(train_start, is_months)
        test_end = add_months(test_start, oos_months) - pd.Timedelta(minutes=15)
        if test_end > last_month + pd.offsets.MonthEnd(1):
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start = add_months(train_start, oos_months)
    return folds


def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index <= end)].copy()


def _rebuild_equity_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"timestamp": [], "equity": []})
    frame = trades.sort_values("exit_time").copy()
    equity = INITIAL_CAPITAL + frame["pnl_usd"].astype(float).cumsum()
    return pd.DataFrame({"timestamp": pd.to_datetime(frame["exit_time"], utc=True), "equity": equity})


def run_walkforward(
    *,
    strategy_name: str,
    strategy_module: Any,
    frame: pd.DataFrame,
    combos: list[dict],
    engine_config: Any,
    news_config: NewsConfig,
    is_months: int,
    oos_months: int,
    precision_package: dict[str, pd.DataFrame] | None = None,
    data_source_used: str | None = None,
) -> WFAResult:
    news_result = load_news_events(engine_config.pair, news_config)
    news_events = news_result.events
    news_filter_used = news_result.enabled
    folds = build_wfa_folds(frame, is_months, oos_months)
    fold_rows: list[dict[str, Any]] = []
    oos_trades_parts: list[pd.DataFrame] = []

    for train_start, train_end, test_start, test_end in folds:
        train_frame = _slice_frame(frame, train_start, train_end)
        test_frame = _slice_frame(frame, test_start, test_end)
        train_precision_package = slice_high_precision_package_to_frame(precision_package, train_frame.index)
        test_precision_package = slice_high_precision_package_to_frame(precision_package, test_frame.index)
        if len(train_frame) <= strategy_module.WARMUP_BARS + 2 or len(test_frame) <= strategy_module.WARMUP_BARS + 2:
            continue

        train_block = build_entry_block(entry_open_index(train_frame.index), news_events, news_config)
        test_block = build_entry_block(entry_open_index(test_frame.index), news_events, news_config)
        best_score = -float("inf")
        best_params: dict[str, Any] | None = None

        for params in combos:
            result = run_backtest(
                strategy_module,
                train_frame,
                params,
                engine_config,
                train_block,
                news_filter_used,
                precision_package=train_precision_package,
                data_source_used=data_source_used,
            )
            summary, *_ = summarize_result(strategy_name, result.trades, result.equity_curve, params, news_filter_used, INITIAL_CAPITAL, None)
            score = score_is_summary(summary)
            if score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            continue

        oos_result = run_backtest(
            strategy_module,
            test_frame,
            best_params,
            engine_config,
            test_block,
            news_filter_used,
            precision_package=test_precision_package,
            data_source_used=data_source_used,
        )
        oos_summary, *_ = summarize_result(strategy_name, oos_result.trades, oos_result.equity_curve, best_params, news_filter_used, INITIAL_CAPITAL, None)
        fold_rows.append(
            {
                "train_period": f"{train_start.date()}->{train_end.date()}",
                "test_period": f"{test_start.date()}->{test_end.date()}",
                "params_used": json.dumps(best_params, ensure_ascii=False),
                "trades": oos_summary["total_trades"],
                "win_rate": oos_summary["win_rate"],
                "pnl_r": float(oos_result.trades["pnl_r"].sum()) if not oos_result.trades.empty else 0.0,
                "pnl_usd": float(oos_result.trades["pnl_usd"].sum()) if not oos_result.trades.empty else 0.0,
                "max_drawdown_pct": oos_summary["max_drawdown_pct"],
                "profit_factor": oos_summary["profit_factor"],
            }
        )
        if not oos_result.trades.empty:
            oos_trades_parts.append(oos_result.trades.copy())

    oos_trades = pd.concat(oos_trades_parts, ignore_index=True).sort_values("exit_time") if oos_trades_parts else pd.DataFrame()
    oos_equity_curve = _rebuild_equity_from_trades(oos_trades)
    oos_summary, *_ = summarize_result(strategy_name, oos_trades, oos_equity_curve, {"wfa_is_months": is_months, "wfa_oos_months": oos_months}, news_filter_used, INITIAL_CAPITAL, None)
    return WFAResult(pd.DataFrame(fold_rows), oos_trades, oos_equity_curve, oos_summary)


def run_default_and_alt_wfa(
    *,
    strategy_name: str,
    strategy_module: Any,
    frame: pd.DataFrame,
    combos: list[dict],
    engine_config: Any,
    news_config: NewsConfig,
    precision_package: dict[str, pd.DataFrame] | None = None,
    data_source_used: str | None = None,
) -> tuple[WFAResult, WFAResult]:
    default_result = run_walkforward(
        strategy_name=strategy_name,
        strategy_module=strategy_module,
        frame=frame,
        combos=combos,
        engine_config=engine_config,
        news_config=news_config,
        is_months=DEFAULT_WFA_IS_MONTHS,
        oos_months=DEFAULT_WFA_OOS_MONTHS,
        precision_package=precision_package,
        data_source_used=data_source_used,
    )
    alt_result = run_walkforward(
        strategy_name=strategy_name,
        strategy_module=strategy_module,
        frame=frame,
        combos=combos,
        engine_config=engine_config,
        news_config=news_config,
        is_months=ALT_WFA_IS_MONTHS,
        oos_months=ALT_WFA_OOS_MONTHS,
        precision_package=precision_package,
        data_source_used=data_source_used,
    )
    return default_result, alt_result
