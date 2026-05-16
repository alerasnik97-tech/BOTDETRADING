from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from research_lab.build_pm_safe_news_dataset import OUTPUT_FILE as PM_SAFE_NEWS_FILE
from research_lab.build_pm_safe_news_dataset import build_pm_safe_news_dataset
from research_lab.config import DEFAULT_DATA_DIRS, EngineConfig, INITIAL_CAPITAL, NY_TZ, NewsConfig, with_execution_mode
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import export_strategy_bundle, summarize_result
from research_lab.strategies import ict_ifvg_repricing_pm


RESULTS_DIR = Path("results") / "ifvg_repricing_pm"
PAIR = "EURUSD"
TIMEFRAME = "M5"
STRATEGY_MODULES = [ict_ifvg_repricing_pm]
PERIODS: dict[str, tuple[str, str]] = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}

SERIOUS_DEV_MIN_TRADES = 24
SERIOUS_DEV_MIN_TRADES_PER_MONTH = 0.50
SERIOUS_DEV_MIN_PF = 1.10
SERIOUS_DEV_MIN_EXPECTANCY = 0.03
SERIOUS_MIN_VAL_HOLD_TRADES = 6
SERIOUS_MIN_VAL_HOLD_PF = 0.95
SERIOUS_FULL_MIN_TRADES = 40


def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_ifvg_repricing_pm"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_engine_config() -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        shock_candle_atr_max=3.0,
        assumed_spread_pips=1.0,
        max_spread_pips=1.6,
        commission_per_lot_roundturn_usd=7.0,
        slippage_pips=0.1,
        max_trades_per_day=1,
        cost_profile="base",
        intrabar_policy="conservative",
        session_cutoff="16:30",
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, "normal_mode")


def build_news_config() -> NewsConfig:
    return NewsConfig(
        enabled=True,
        file_path=PM_SAFE_NEWS_FILE,
        raw_file_path=PM_SAFE_NEWS_FILE,
        source_approved=True,
        pre_minutes=45,
        post_minutes=90,
        currencies=("USD",),
        impact_levels=("HIGH",),
    )


def minute_to_hhmm(value: int) -> str:
    hour = int(value) // 60
    minute = int(value) % 60
    return f"{hour:02d}:{minute:02d}"


def schedule_used_from_params(params: dict[str, Any]) -> dict[str, str]:
    return {
        "entry_start": minute_to_hhmm(int(params["entry_minute_floor"])),
        "entry_end": minute_to_hhmm(int(params["latest_signal_minute"])),
        "force_close": "16:30",
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def news_metrics_from_summary(summary: dict[str, Any], trades_export: pd.DataFrame) -> dict[str, Any]:
    news_exit_count = int((trades_export["exit_reason"] == "news_fortress_kill").sum()) if not trades_export.empty else 0
    return {**summary, "news_exit_count": news_exit_count}


def evaluate_period(
    *,
    strategy_module: Any,
    frame: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: NewsConfig,
    data_source_used: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    period_frame = period_slice(frame, start, end)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        strategy_module,
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
        strategy_module.NAME,
        result.trades,
        result.equity_curve,
        params,
        news_result.enabled,
        INITIAL_CAPITAL,
        None,
        costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
        timeframe=TIMEFRAME,
        schedule_used=schedule_used_from_params(params),
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
    score += float(summary["total_return_pct"]) * 1.1
    score -= float(summary["max_drawdown_pct"]) * 3.2
    score -= float(summary["negative_years"]) * 30.0
    score -= float(summary["news_exit_count"]) * 5.0
    if bool(summary.get("insufficient_sample", False)):
        score -= 60.0
    total_trades = int(summary["total_trades"])
    avg_trades_per_month = float(summary["avg_trades_per_month"])
    if total_trades < 5:
        score -= 1500.0
    elif total_trades < 10:
        score -= 900.0
    elif total_trades < SERIOUS_DEV_MIN_TRADES:
        score -= 240.0
    if avg_trades_per_month < SERIOUS_DEV_MIN_TRADES_PER_MONTH:
        score -= 140.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 80.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 80.0
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
        and int(row["full_total_trades"]) >= SERIOUS_FULL_MIN_TRADES
    )


def verdict_from_selected(selected_row: pd.Series, serious_candidates: int) -> str:
    if serious_candidates > 0 and bool(selected_row["serious_candidate"]):
        return "Promising candidate"
    return "Close the line"


def main() -> None:
    build_pm_safe_news_dataset()
    output_root = build_output_root()
    engine_config = build_engine_config()
    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context="ifvg_line_runner")
    bundle = load_backtest_data_bundle(
        PAIR,
        list(DEFAULT_DATA_DIRS),
        *PERIODS["full_2020_2025"],
        engine_config.execution_mode,
        target_timeframe=TIMEFRAME,
    )
    full_frame = bundle.frame.copy()

    strategy_rows: list[dict[str, Any]] = []

    for strategy_module in STRATEGY_MODULES:
        ranking_rows: list[dict[str, Any]] = []
        for combo_id, params in enumerate(strategy_module.parameter_grid(), start=1):
            dev = evaluate_period(
                strategy_module=strategy_module,
                frame=full_frame,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                data_source_used=bundle.data_source_used,
                start=PERIODS["development_2020_2023"][0],
                end=PERIODS["development_2020_2023"][1],
            )["summary"]
            val = evaluate_period(
                strategy_module=strategy_module,
                frame=full_frame,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                data_source_used=bundle.data_source_used,
                start=PERIODS["validation_2024"][0],
                end=PERIODS["validation_2024"][1],
            )["summary"]
            hold = evaluate_period(
                strategy_module=strategy_module,
                frame=full_frame,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                data_source_used=bundle.data_source_used,
                start=PERIODS["holdout_2025"][0],
                end=PERIODS["holdout_2025"][1],
            )["summary"]
            full = evaluate_period(
                strategy_module=strategy_module,
                frame=full_frame,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                data_source_used=bundle.data_source_used,
                start=PERIODS["full_2020_2025"][0],
                end=PERIODS["full_2020_2025"][1],
            )["summary"]
            ranking_rows.append(
                {
                    "strategy_name": strategy_module.NAME,
                    "combo_id": combo_id,
                    "variant_label": str(params.get("variant_label", f"combo_{combo_id}")),
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "selection_score_dev": selection_score(dev),
                    "serious_candidate": False,
                    "dev_profit_factor": dev["profit_factor"],
                    "dev_expectancy_r": dev["expectancy_r"],
                    "dev_max_drawdown_pct": dev["max_drawdown_pct"],
                    "dev_total_trades": dev["total_trades"],
                    "dev_avg_trades_per_month": dev["avg_trades_per_month"],
                    "val_profit_factor": val["profit_factor"],
                    "val_expectancy_r": val["expectancy_r"],
                    "val_total_trades": val["total_trades"],
                    "hold_profit_factor": hold["profit_factor"],
                    "hold_expectancy_r": hold["expectancy_r"],
                    "hold_total_trades": hold["total_trades"],
                    "full_profit_factor": full["profit_factor"],
                    "full_expectancy_r": full["expectancy_r"],
                    "full_total_trades": full["total_trades"],
                    "full_max_drawdown_pct": full["max_drawdown_pct"],
                }
            )

        ranking = pd.DataFrame(ranking_rows)
        ranking["serious_candidate"] = ranking.apply(serious_gate_from_row, axis=1)
        ranking = ranking.sort_values(["serious_candidate", "selection_score_dev"], ascending=[False, False]).reset_index(drop=True)
        selected_row = ranking.iloc[0].copy()
        selected_params = json.loads(str(selected_row["params_json"]))

        period_bundles: dict[str, dict[str, Any]] = {}
        for label, (start, end) in PERIODS.items():
            period_bundles[label] = evaluate_period(
                strategy_module=strategy_module,
                frame=full_frame,
                params=selected_params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                data_source_used=bundle.data_source_used,
                start=start,
                end=end,
            )

        strategy_dir = output_root / strategy_module.NAME
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
                "period_summaries.json": {label: bundle_["summary"] for label, bundle_ in period_bundles.items()},
                "serious_gate.json": {
                    "dev_total_trades": SERIOUS_DEV_MIN_TRADES,
                    "dev_avg_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                    "dev_profit_factor": SERIOUS_DEV_MIN_PF,
                    "dev_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                    "val_hold_trades": SERIOUS_MIN_VAL_HOLD_TRADES,
                    "val_hold_profit_factor": SERIOUS_MIN_VAL_HOLD_PF,
                    "full_total_trades": SERIOUS_FULL_MIN_TRADES,
                },
                "verdict.json": {"verdict": verdict_from_selected(selected_row, int(ranking["serious_candidate"].sum()))},
            },
        )

        strategy_rows.append(
            {
                "strategy_name": strategy_module.NAME,
                "selected_combo_id": int(selected_row["combo_id"]),
                "variant_label": str(selected_row["variant_label"]),
                "verdict": verdict_from_selected(selected_row, int(ranking["serious_candidate"].sum())),
                "serious_candidates": int(ranking["serious_candidate"].sum()),
                "dev_trades": int(period_bundles["development_2020_2023"]["summary"]["total_trades"]),
                "dev_pf": float(period_bundles["development_2020_2023"]["summary"]["profit_factor"]),
                "dev_expectancy_r": float(period_bundles["development_2020_2023"]["summary"]["expectancy_r"]),
                "val_trades": int(period_bundles["validation_2024"]["summary"]["total_trades"]),
                "val_pf": float(period_bundles["validation_2024"]["summary"]["profit_factor"]),
                "val_expectancy_r": float(period_bundles["validation_2024"]["summary"]["expectancy_r"]),
                "hold_trades": int(period_bundles["holdout_2025"]["summary"]["total_trades"]),
                "hold_pf": float(period_bundles["holdout_2025"]["summary"]["profit_factor"]),
                "hold_expectancy_r": float(period_bundles["holdout_2025"]["summary"]["expectancy_r"]),
                "full_trades": int(period_bundles["full_2020_2025"]["summary"]["total_trades"]),
                "full_pf": float(period_bundles["full_2020_2025"]["summary"]["profit_factor"]),
                "full_expectancy_r": float(period_bundles["full_2020_2025"]["summary"]["expectancy_r"]),
                "full_max_drawdown_pct": float(period_bundles["full_2020_2025"]["summary"]["max_drawdown_pct"]),
                "news_exit_count": int(period_bundles["full_2020_2025"]["summary"]["news_exit_count"]),
                "selected_params_json": json.dumps(selected_params, ensure_ascii=False, sort_keys=True),
            }
        )

    strategy_scorecard = pd.DataFrame(strategy_rows).sort_values(["full_pf", "full_expectancy_r"], ascending=[False, False]).reset_index(drop=True)
    strategy_scorecard.to_csv(output_root / "ifvg_strategy_scorecard.csv", index=False)
    manifest = {
        "pair": PAIR,
        "timeframe": TIMEFRAME,
        "execution_mode": engine_config.execution_mode,
        "data_source_used": bundle.data_source_used,
        "news_dataset": str(PM_SAFE_NEWS_FILE),
        "output_root": str(output_root),
        "strategies": [module.NAME for module in STRATEGY_MODULES],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(output_root))


if __name__ == "__main__":
    main()
