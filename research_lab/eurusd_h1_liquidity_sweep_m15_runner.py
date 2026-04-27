from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from research_lab.config import (
    EngineConfig,
    INITIAL_CAPITAL,
    NY_TZ,
    canonical_news_config,
    canonical_prepared_data_dirs,
    with_execution_mode,
)
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import export_strategy_bundle, summarize_result
from research_lab.strategies import eurusd_h1_liquidity_sweep_m15


RESULTS_DIR = Path("results") / "eurusd_h1_liquidity_sweep_m15"
PAIR = "EURUSD"
TIMEFRAME = "M15"
PERIODS: dict[str, tuple[str, str]] = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}

SERIOUS_DEV_MIN_TRADES = 36
SERIOUS_DEV_MIN_TRADES_PER_MONTH = 0.75
SERIOUS_DEV_MIN_PF = 1.10
SERIOUS_DEV_MIN_EXPECTANCY = 0.03
SERIOUS_MIN_VAL_HOLD_TRADES = 10
SERIOUS_MIN_VAL_HOLD_PF = 0.95
SERIOUS_MIN_VAL_HOLD_EXPECTANCY = 0.0
SERIOUS_FULL_MIN_TRADES = 54


def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_{eurusd_h1_liquidity_sweep_m15.NAME}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_engine_config() -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        shock_candle_atr_max=3.0,
        assumed_spread_pips=1.2,
        max_spread_pips=1.8,
        commission_per_lot_roundturn_usd=7.0,
        slippage_pips=0.2,
        max_trades_per_day=1,
        cost_profile="base",
        intrabar_policy="conservative",
        session_cutoff="16:30",
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, "normal_mode")


def build_news_config():
    return canonical_news_config(
        PAIR,
        enabled=True,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
    )


def schedule_used() -> dict[str, str]:
    return {
        "entry_start": "08:00",
        "entry_end": "16:30",
        "force_close": "16:30",
        "target_frame": TIMEFRAME,
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def news_metrics_from_summary(summary: dict[str, Any], trades_export: pd.DataFrame) -> dict[str, Any]:
    news_exit_count = int((trades_export["exit_reason"] == "news_fortress_kill").sum()) if not trades_export.empty else 0
    return {**summary, "news_exit_count": news_exit_count}


def evaluate_period(
    *,
    frame: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: Any,
    data_source_used: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    period_frame = period_slice(frame, start, end)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        eurusd_h1_liquidity_sweep_m15,
        period_frame,
        params,
        engine_config,
        news_block,
        news_result.enabled,
        precision_package=None,
        data_source_used=data_source_used,
        news_events=news_result.events,
        news_settings=news_config,
    )
    summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
        eurusd_h1_liquidity_sweep_m15.NAME,
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
    score += pf_for_score * 160.0
    score += float(summary["expectancy_r"]) * 1200.0
    score += float(summary["total_return_pct"]) * 1.0
    score -= float(summary["max_drawdown_pct"]) * 3.0
    score -= float(summary["negative_years"]) * 30.0
    score -= float(summary["news_exit_count"]) * 5.0
    if bool(summary.get("insufficient_sample", False)):
        score -= 100.0
    total_trades = int(summary["total_trades"])
    avg_trades_per_month = float(summary["avg_trades_per_month"])
    if total_trades < 10:
        score -= 1200.0
    elif total_trades < SERIOUS_DEV_MIN_TRADES:
        score -= 250.0
    if avg_trades_per_month < SERIOUS_DEV_MIN_TRADES_PER_MONTH:
        score -= 180.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 120.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 120.0
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
    if float(selected_row["full_total_trades"]) >= SERIOUS_FULL_MIN_TRADES and float(selected_row["full_profit_factor"]) > 1.0:
        return "Still diagnostic / not defendable"
    return "Close the line"


def main() -> Path:
    output_root = build_output_root()
    engine_config = build_engine_config()
    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context="eurusd_h1_liquidity_sweep_m15_runner")
    bundle = load_backtest_data_bundle(
        PAIR,
        list(canonical_prepared_data_dirs(PAIR)),
        *PERIODS["full_2020_2025"],
        engine_config.execution_mode,
        target_timeframe=TIMEFRAME,
    )
    full_frame = bundle.frame.copy()

    ranking_rows: list[dict[str, Any]] = []
    period_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for combo_id, params in enumerate(eurusd_h1_liquidity_sweep_m15.parameter_grid(), start=1):
        period_results: dict[str, dict[str, Any]] = {}
        for label, (start, end) in PERIODS.items():
            period_results[label] = evaluate_period(
                frame=full_frame,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                data_source_used=bundle.data_source_used,
                start=start,
                end=end,
            )

        dev = period_results["development_2020_2023"]["summary"]
        val = period_results["validation_2024"]["summary"]
        hold = period_results["holdout_2025"]["summary"]
        full = period_results["full_2020_2025"]["summary"]
        params_json = json.dumps(params, sort_keys=True)
        period_cache[params_json] = period_results
        ranking_rows.append(
            {
                "combo_id": combo_id,
                "variant_label": params.get("variant_label", f"combo_{combo_id}"),
                "params_json": params_json,
                "dev_total_trades": dev["total_trades"],
                "dev_avg_trades_per_month": dev["avg_trades_per_month"],
                "dev_profit_factor": dev["profit_factor"],
                "dev_expectancy_r": dev["expectancy_r"],
                "val_total_trades": val["total_trades"],
                "val_profit_factor": val["profit_factor"],
                "val_expectancy_r": val["expectancy_r"],
                "hold_total_trades": hold["total_trades"],
                "hold_profit_factor": hold["profit_factor"],
                "hold_expectancy_r": hold["expectancy_r"],
                "full_total_trades": full["total_trades"],
                "full_avg_trades_per_month": full["avg_trades_per_month"],
                "full_profit_factor": full["profit_factor"],
                "full_expectancy_r": full["expectancy_r"],
                "full_max_drawdown_pct": full["max_drawdown_pct"],
                "full_total_return_pct": full["total_return_pct"],
                "selection_score": selection_score(dev),
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    ranking["serious_candidate"] = ranking.apply(serious_gate_from_row, axis=1)
    serious_candidates = int(ranking["serious_candidate"].sum())
    ranking = ranking.sort_values(["selection_score", "full_profit_factor"], ascending=[False, False]).reset_index(drop=True)
    selected_row = ranking.iloc[0]
    selected_params = json.loads(selected_row["params_json"])
    selected_periods = period_cache[selected_row["params_json"]]
    full_result = selected_periods["full_2020_2025"]
    verdict = verdict_from_selected(selected_row, serious_candidates)

    strategy_dir = output_root / eurusd_h1_liquidity_sweep_m15.NAME
    period_summaries = {label: payload["summary"] for label, payload in selected_periods.items()}
    export_strategy_bundle(
        strategy_dir,
        summary=full_result["summary"],
        trades_export=full_result["trades_export"],
        monthly_stats=full_result["monthly_stats"],
        yearly_stats=full_result["yearly_stats"],
        equity_export=full_result["equity_export"],
        optimization_results=ranking,
        extra_json={
            "selected_params.json": selected_params,
            "period_summaries.json": period_summaries,
            "serious_gate.json": {
                "development": {
                    "min_trades": SERIOUS_DEV_MIN_TRADES,
                    "min_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                    "min_profit_factor": SERIOUS_DEV_MIN_PF,
                    "min_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                },
                "validation_holdout": {
                    "min_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                    "min_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                    "min_expectancy_r": SERIOUS_MIN_VAL_HOLD_EXPECTANCY,
                },
                "full_sample": {
                    "min_trades": SERIOUS_FULL_MIN_TRADES,
                },
            },
            "verdict.json": {"verdict": verdict, "serious_candidates": serious_candidates},
        },
    )
    ranking.to_csv(output_root / "eurusd_h1_m15_scorecard.csv", index=False)
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "strategy": eurusd_h1_liquidity_sweep_m15.NAME,
                "pair": PAIR,
                "timeframe": TIMEFRAME,
                "periods": PERIODS,
                "engine_config": {
                    "execution_mode": engine_config.execution_mode,
                    "cost_profile": engine_config.cost_profile,
                    "session_cutoff": engine_config.session_cutoff,
                },
                "schedule_used": schedule_used(),
                "news_dataset": str(news_result.final_dataset_path),
                "news_enabled": news_result.enabled,
                "selected_params": selected_params,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_root


if __name__ == "__main__":
    main()
