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
    atr,
    fx_market_mask,
    load_high_precision_package,
    validate_price_frame,
)
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import export_strategy_bundle, summarize_result
from research_lab.strategies import am_opening_range_expansion_retest


RESULTS_DIR = Path("results") / "am_opening_range_expansion_retest"
PAIR = "EURUSD"
TIMEFRAME = "M1"
FRAME_BUILD_START = "09:30"
FRAME_BUILD_END = "11:30"
OPENING_RANGE_START = "09:30"
OPENING_RANGE_END = "09:44"
BREAKOUT_START = "09:45"
BREAKOUT_END = "10:30"
ENTRY_START = "09:46"
ENTRY_END = "10:30"
PERIODS: dict[str, tuple[str, str]] = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}

SERIOUS_DEV_MIN_TRADES = 48
SERIOUS_DEV_MIN_TRADES_PER_MONTH = 1.0
SERIOUS_DEV_MIN_PF = 1.10
SERIOUS_DEV_MIN_EXPECTANCY = 0.03
SERIOUS_MIN_VAL_HOLD_TRADES = 12
SERIOUS_MIN_VAL_HOLD_PF = 0.95
SERIOUS_MIN_VAL_HOLD_EXPECTANCY = 0.0
SERIOUS_FULL_MIN_TRADES = 72
PIP_SIZE = 0.0001


def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_{am_opening_range_expansion_retest.NAME}"
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
        session_cutoff="11:30",
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
        "frame_build_window": f"{FRAME_BUILD_START}-{FRAME_BUILD_END}",
        "opening_range_window": f"{OPENING_RANGE_START}-{OPENING_RANGE_END}",
        "breakout_window": f"{BREAKOUT_START}-{BREAKOUT_END}",
        "entry_start": ENTRY_START,
        "entry_end": ENTRY_END,
        "force_close": "11:30",
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


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


def _local_dates(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index=index, data=index.tz_convert(NY_TZ).date)


def _ny_vwap(frame: pd.DataFrame) -> pd.Series:
    tp = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    cum_tpv = (tp * frame["volume"]).cumsum()
    cum_v = frame["volume"].replace(0.0, np.nan).cumsum()
    return cum_tpv / cum_v


def _close_location(frame: pd.DataFrame) -> pd.Series:
    denom = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    return (frame["close"] - frame["low"]) / denom


def _annotate_opening_range_day(day: pd.DataFrame) -> pd.DataFrame:
    research = day.between_time(FRAME_BUILD_START, FRAME_BUILD_END).copy()
    opening_range = research.between_time(OPENING_RANGE_START, OPENING_RANGE_END).copy()
    if opening_range.empty or len(opening_range) < 10:
        return research.iloc[0:0].copy()

    or_high = float(opening_range["high"].max())
    or_low = float(opening_range["low"].min())
    or_range = or_high - or_low
    atr_ref = float(opening_range["atr14"].dropna().iloc[-1]) if not opening_range["atr14"].dropna().empty else np.nan
    if not math.isfinite(atr_ref) or atr_ref <= 0 or or_range <= 0:
        return research.iloc[0:0].copy()

    or_range_pips = or_range / PIP_SIZE
    or_range_atr = or_range / atr_ref
    if or_range_pips < 3.0 or or_range_atr < 0.6:
        return research.iloc[0:0].copy()

    or_mid = (or_high + or_low) / 2.0
    research["ore_or_high"] = or_high
    research["ore_or_low"] = or_low
    research["ore_or_mid"] = or_mid
    research["ore_or_range_pips"] = or_range_pips
    research["ore_or_range_atr"] = or_range_atr
    research["ore_ny_vwap"] = _ny_vwap(research)
    research["ore_stop_price_long"] = or_mid - PIP_SIZE
    research["ore_stop_price_short"] = or_mid + PIP_SIZE

    breakout_window = research.between_time(BREAKOUT_START, BREAKOUT_END).index
    breakout_mask = research.index.isin(breakout_window)
    close_loc = research["close_loc"].fillna(0.5)
    bull_break = (
        breakout_mask
        & (research["close"] > (or_high + PIP_SIZE))
        & (research["range_atr"] >= 1.1)
        & (close_loc >= 0.70)
    )
    bear_break = (
        breakout_mask
        & (research["close"] < (or_low - PIP_SIZE))
        & (research["range_atr"] >= 1.1)
        & (close_loc <= 0.30)
    )
    outside_long = (research["close"] > or_high) & (research["close"] > research["ore_ny_vwap"])
    outside_short = (research["close"] < or_low) & (research["close"] < research["ore_ny_vwap"])
    recent_bull_break = bull_break.shift(1, fill_value=False) | bull_break.shift(2, fill_value=False)
    recent_bear_break = bear_break.shift(1, fill_value=False) | bear_break.shift(2, fill_value=False)

    research["ore_breakout_long"] = bull_break
    research["ore_breakout_short"] = bear_break
    research["ore_acceptance_long"] = breakout_mask & outside_long & outside_long.shift(1, fill_value=False) & recent_bull_break
    research["ore_acceptance_short"] = breakout_mask & outside_short & outside_short.shift(1, fill_value=False) & recent_bear_break
    return research


def _build_m1_research_frame(mid_m1: pd.DataFrame) -> pd.DataFrame:
    base = mid_m1.copy()
    base["local_date"] = _local_dates(base.index).to_numpy()
    base["atr14"] = atr(base, 14)
    base["bar_range"] = base["high"] - base["low"]
    base["range_atr"] = base["bar_range"] / base["atr14"].replace(0.0, np.nan)
    base["close_loc"] = _close_location(base)

    research = base.between_time(FRAME_BUILD_START, FRAME_BUILD_END).copy()
    day_frames: list[pd.DataFrame] = []
    for _, day in research.groupby("local_date", sort=False):
        annotated = _annotate_opening_range_day(day)
        if not annotated.empty:
            day_frames.append(annotated)
    if not day_frames:
        return research.iloc[0:0].copy()
    frame = pd.concat(day_frames).sort_index()
    return frame.dropna(
        subset=[
            "atr14",
            "range_atr",
            "close_loc",
            "ore_or_high",
            "ore_or_low",
            "ore_or_mid",
            "ore_ny_vwap",
            "ore_stop_price_long",
            "ore_stop_price_short",
        ]
    ).copy()


def build_research_frame(start: str, end: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    filtered = _filtered_high_precision_package(start, end)
    frame = _build_m1_research_frame(filtered["mid_m1"])
    return frame, filtered


def align_precision_package(filtered: dict[str, pd.DataFrame], frame_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    bid_exec = filtered["bid_m1"].loc[frame_index].copy()
    ask_exec = filtered["ask_m1"].loc[frame_index].copy()
    mid_exec = filtered["mid_m1"].loc[frame_index].copy()
    return {
        "bid_m1": bid_exec.copy(),
        "ask_m1": ask_exec.copy(),
        "mid_m1": mid_exec.copy(),
        "bid_exec": bid_exec.copy(),
        "ask_exec": ask_exec.copy(),
        "mid_exec": mid_exec.copy(),
        "bid_m15": bid_exec.copy(),
        "ask_m15": ask_exec.copy(),
        "mid_m15": mid_exec.copy(),
    }


def news_metrics_from_summary(summary: dict[str, Any], trades_export: pd.DataFrame) -> dict[str, Any]:
    news_exit_count = int((trades_export["exit_reason"] == "news_fortress_kill").sum()) if not trades_export.empty else 0
    return {**summary, "news_exit_count": news_exit_count}


def evaluate_period(
    *,
    frame: pd.DataFrame,
    filtered_precision: dict[str, pd.DataFrame],
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
    period_precision = align_precision_package(filtered_precision, period_frame.index)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        strategy_module=am_opening_range_expansion_retest,
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
        am_opening_range_expansion_retest.NAME,
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
    return {
        "summary": news_metrics_from_summary(summary, trades_export),
        "trades_export": trades_export,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
    }


def selection_score(summary: dict[str, Any]) -> float:
    profit_factor = float(summary["profit_factor"])
    pf_for_score = 3.0 if not math.isfinite(profit_factor) else min(profit_factor, 3.0)
    score = 0.0
    score += pf_for_score * 140.0
    score += float(summary["expectancy_r"]) * 1100.0
    score += float(summary["total_return_pct"]) * 1.0
    score -= float(summary["max_drawdown_pct"]) * 3.0
    score -= float(summary["negative_years"]) * 35.0
    score -= float(summary["news_exit_count"]) * 5.0
    if bool(summary.get("insufficient_sample", False)):
        score -= 80.0
    total_trades = int(summary["total_trades"])
    avg_trades_per_month = float(summary["avg_trades_per_month"])
    if total_trades < 12:
        score -= 800.0
    elif total_trades < SERIOUS_DEV_MIN_TRADES:
        score -= 180.0
    if avg_trades_per_month < SERIOUS_DEV_MIN_TRADES_PER_MONTH:
        score -= 120.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 100.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 100.0
    return score


def serious_gate_from_row(row: pd.Series) -> bool:
    return (
        int(row["dev_total_trades"]) >= SERIOUS_DEV_MIN_TRADES
        and float(row["dev_avg_trades_per_month"]) >= SERIOUS_DEV_MIN_TRADES_PER_MONTH
        and float(row["dev_profit_factor"]) >= SERIOUS_DEV_MIN_PF
        and float(row["dev_expectancy_r"]) >= SERIOUS_DEV_MIN_EXPECTANCY
        and int(row["val_total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and int(row["hold_total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and float(row["val_profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(row["hold_profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(row["val_expectancy_r"]) >= SERIOUS_MIN_VAL_HOLD_EXPECTANCY
        and float(row["hold_expectancy_r"]) >= SERIOUS_MIN_VAL_HOLD_EXPECTANCY
        and int(row["full_total_trades"]) >= SERIOUS_FULL_MIN_TRADES
    )


def verdict_from_selected(selected_row: pd.Series, serious_candidates: int) -> str:
    if serious_candidates > 0 and bool(selected_row["serious_candidate"]):
        return "Promising candidate"
    if float(selected_row["full_total_trades"]) >= 36 and float(selected_row["full_profit_factor"]) > 1.0:
        return "Still diagnostic / not defendable"
    return "Close the line"


def main() -> Path:
    am_summary = build_am_grade_news_dataset()
    if am_summary["module_verdict"] != "READY_FOR_STRICT_AM_RESEARCH":
        raise RuntimeError(
            "La linea AM Opening Range Expansion Retest requiere compuerta AM aprobada. "
            f"Veredicto actual={am_summary['module_verdict']}."
        )

    output_root = build_output_root()
    engine_config = build_engine_config()
    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context="am_opening_range_expansion_retest_runner")
    full_frame, filtered_precision = build_research_frame(*PERIODS["full_2020_2025"])

    params = am_opening_range_expansion_retest.default_params()
    summaries: dict[str, dict[str, Any]] = {}
    period_bundles: dict[str, dict[str, Any]] = {}
    for label, (start, end) in PERIODS.items():
        bundle = evaluate_period(
            frame=full_frame,
            filtered_precision=filtered_precision,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=start,
            end=end,
        )
        period_bundles[label] = bundle
        summaries[label] = bundle["summary"]

    ranking = pd.DataFrame(
        [
            {
                "strategy_name": am_opening_range_expansion_retest.NAME,
                "combo_id": 1,
                "variant_label": str(params["variant_label"]),
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "selection_score_dev": selection_score(summaries["development_2020_2023"]),
                "serious_candidate": False,
                "dev_profit_factor": summaries["development_2020_2023"]["profit_factor"],
                "dev_expectancy_r": summaries["development_2020_2023"]["expectancy_r"],
                "dev_max_drawdown_pct": summaries["development_2020_2023"]["max_drawdown_pct"],
                "dev_total_trades": summaries["development_2020_2023"]["total_trades"],
                "dev_avg_trades_per_month": summaries["development_2020_2023"]["avg_trades_per_month"],
                "val_profit_factor": summaries["validation_2024"]["profit_factor"],
                "val_expectancy_r": summaries["validation_2024"]["expectancy_r"],
                "val_total_trades": summaries["validation_2024"]["total_trades"],
                "hold_profit_factor": summaries["holdout_2025"]["profit_factor"],
                "hold_expectancy_r": summaries["holdout_2025"]["expectancy_r"],
                "hold_total_trades": summaries["holdout_2025"]["total_trades"],
                "full_profit_factor": summaries["full_2020_2025"]["profit_factor"],
                "full_expectancy_r": summaries["full_2020_2025"]["expectancy_r"],
                "full_total_trades": summaries["full_2020_2025"]["total_trades"],
                "full_max_drawdown_pct": summaries["full_2020_2025"]["max_drawdown_pct"],
            }
        ]
    )
    ranking["serious_candidate"] = ranking.apply(serious_gate_from_row, axis=1)
    selected_row = ranking.iloc[0].copy()
    selected_params = json.loads(str(selected_row["params_json"]))
    verdict = verdict_from_selected(selected_row, int(ranking["serious_candidate"].sum()))

    strategy_dir = output_root / am_opening_range_expansion_retest.NAME
    strategy_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(strategy_dir / "ranking.csv", index=False)
    export_strategy_bundle(
        strategy_dir,
        summary=period_bundles["full_2020_2025"]["summary"],
        trades_export=period_bundles["full_2020_2025"]["trades_export"],
        monthly_stats=period_bundles["full_2020_2025"]["monthly_stats"],
        yearly_stats=period_bundles["full_2020_2025"]["yearly_stats"],
        equity_export=period_bundles["full_2020_2025"]["equity_export"],
        optimization_results=ranking,
        extra_json={
            "selected_params.json": selected_params,
            "period_summaries.json": summaries,
            "serious_gate.json": {
                "dev_total_trades": SERIOUS_DEV_MIN_TRADES,
                "dev_avg_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                "dev_profit_factor": SERIOUS_DEV_MIN_PF,
                "dev_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                "val_hold_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                "val_hold_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                "val_hold_expectancy_r": SERIOUS_MIN_VAL_HOLD_EXPECTANCY,
                "full_total_trades": SERIOUS_FULL_MIN_TRADES,
            },
            "frame_contract.json": {
                "canonical_timeframe": TIMEFRAME,
                "frame_build_window": f"{FRAME_BUILD_START}-{FRAME_BUILD_END}",
                "opening_range_window": f"{OPENING_RANGE_START}-{OPENING_RANGE_END}",
                "breakout_window": f"{BREAKOUT_START}-{BREAKOUT_END}",
                "entry_window": f"{ENTRY_START}-{ENTRY_END}",
                "signal_family": "opening_range_expansion_retest",
            },
            "verdict.json": {"verdict": verdict},
        },
    )

    scorecard = pd.DataFrame(
        [
            {
                "strategy_name": am_opening_range_expansion_retest.NAME,
                "selected_combo_id": int(selected_row["combo_id"]),
                "variant_label": str(selected_row["variant_label"]),
                "verdict": verdict,
                "serious_candidates": int(ranking["serious_candidate"].sum()),
                "dev_trades": int(summaries["development_2020_2023"]["total_trades"]),
                "dev_pf": float(summaries["development_2020_2023"]["profit_factor"]),
                "dev_expectancy_r": float(summaries["development_2020_2023"]["expectancy_r"]),
                "val_trades": int(summaries["validation_2024"]["total_trades"]),
                "val_pf": float(summaries["validation_2024"]["profit_factor"]),
                "val_expectancy_r": float(summaries["validation_2024"]["expectancy_r"]),
                "hold_trades": int(summaries["holdout_2025"]["total_trades"]),
                "hold_pf": float(summaries["holdout_2025"]["profit_factor"]),
                "hold_expectancy_r": float(summaries["holdout_2025"]["expectancy_r"]),
                "full_trades": int(summaries["full_2020_2025"]["total_trades"]),
                "full_pf": float(summaries["full_2020_2025"]["profit_factor"]),
                "full_expectancy_r": float(summaries["full_2020_2025"]["expectancy_r"]),
                "full_max_drawdown_pct": float(summaries["full_2020_2025"]["max_drawdown_pct"]),
                "news_exit_count": int(summaries["full_2020_2025"]["news_exit_count"]),
                "selected_params_json": json.dumps(selected_params, ensure_ascii=False, sort_keys=True),
            }
        ]
    )
    scorecard.to_csv(output_root / "am_strategy_scorecard.csv", index=False)
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "pair": PAIR,
                "timeframe": TIMEFRAME,
                "periods": PERIODS,
                "news_file": str(AM_NEWS_FILE),
                "schedule": schedule_used(),
                "engine": {
                    "risk_pct": engine_config.risk_pct,
                    "max_spread_pips": engine_config.max_spread_pips,
                    "slippage_pips": engine_config.slippage_pips,
                    "session_cutoff": engine_config.session_cutoff,
                    "hard_stop_required": engine_config.enforce_hard_stop,
                },
                "serious_gate": {
                    "dev_total_trades": SERIOUS_DEV_MIN_TRADES,
                    "dev_avg_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                    "dev_profit_factor": SERIOUS_DEV_MIN_PF,
                    "dev_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                    "val_hold_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                    "val_hold_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                    "val_hold_expectancy_r": SERIOUS_MIN_VAL_HOLD_EXPECTANCY,
                    "full_total_trades": SERIOUS_FULL_MIN_TRADES,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(scorecard.to_string(index=False))
    return output_root


if __name__ == "__main__":
    main()
