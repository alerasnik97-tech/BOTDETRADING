from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_lab.build_pm_safe_news_dataset import OUTPUT_FILE as PM_SAFE_NEWS_FILE
from research_lab.build_pm_safe_news_dataset import OUTPUT_SUMMARY_FILE as PM_SAFE_NEWS_SUMMARY_FILE
from research_lab.build_pm_safe_news_dataset import build_pm_safe_news_dataset
from research_lab.config import EngineConfig, INITIAL_CAPITAL, NY_TZ, NewsConfig, with_execution_mode
from research_lab.data_loader import (
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    fx_market_mask,
    load_high_precision_package,
    prepare_common_frame,
    slice_high_precision_package_to_frame,
    validate_price_frame,
)
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import export_strategy_bundle, summarize_result
from research_lab.strategies import pm_micro_reclaim_m3 as strategy_module


RESULTS_DIR = Path("results") / "pm_micro_reclaim_m3"
PAIR = "EURUSD"
SERIOUS_DEV_MIN_TRADES = 18
SERIOUS_DEV_MIN_TRADES_PER_MONTH = 0.35
SERIOUS_DEV_MIN_PF = 1.05
SERIOUS_DEV_MIN_EXPECTANCY = 0.03
SERIOUS_MIN_VAL_HOLD_TRADES = 3
SERIOUS_MIN_VAL_HOLD_PF = 0.95
PERIODS: dict[str, tuple[str, str]] = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
    "full_2020_2025": ("2020-01-01", "2025-12-31"),
}
def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_pm_micro_reclaim_m3"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def load_m3_precision_context(pair: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    package = load_high_precision_package(pair, DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    filtered: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame.loc[fx_market_mask(frame.index)].copy()
        validate_price_frame(frame)
        filtered[f"{side}_m1"] = frame

    bid_m3 = _resample_ohlcv(filtered["bid_m1"], "3min")
    ask_m3 = _resample_ohlcv(filtered["ask_m1"], "3min")
    mid_m3 = _resample_ohlcv(filtered["mid_m1"], "3min")
    strategy_frame = prepare_common_frame(mid_m3, target_timeframe="M3")
    common_index = strategy_frame.index.intersection(bid_m3.index).intersection(ask_m3.index)
    if common_index.empty:
        raise ValueError("No se pudo alinear el contexto M3 con BID/ASK M1.")

    aligned_package = {
        "bid_m1": filtered["bid_m1"],
        "ask_m1": filtered["ask_m1"],
        "mid_m1": filtered["mid_m1"],
        "bid_m15": bid_m3.loc[common_index].copy(),
        "ask_m15": ask_m3.loc[common_index].copy(),
        "mid_m15": mid_m3.loc[common_index].copy(),
    }
    return strategy_frame.loc[common_index].copy(), aligned_package


def build_engine_config() -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        shock_candle_atr_max=3.0,
        assumed_spread_pips=1.0,
        max_spread_pips=1.4,
        commission_per_lot_roundturn_usd=7.0,
        slippage_pips=0.1,
        max_trades_per_day=1,
        cost_profile="precision",
        intrabar_policy="conservative",
        session_cutoff="16:00",
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, "high_precision_mode")


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
        "force_close": "16:00",
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=3)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def news_metrics_from_summary(summary: dict[str, Any], trades_export: pd.DataFrame) -> dict[str, Any]:
    news_exit_count = int((trades_export["exit_reason"] == "news_fortress_kill").sum()) if not trades_export.empty else 0
    return {
        **summary,
        "news_exit_count": news_exit_count,
        "years_positive": int(summary.get("years_positive", 0)),
    }


def evaluate_period(
    *,
    frame: pd.DataFrame,
    precision_package: dict[str, pd.DataFrame],
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: NewsConfig,
    start: str,
    end: str,
) -> dict[str, Any]:
    period_frame = period_slice(frame, start, end)
    period_precision = slice_high_precision_package_to_frame(precision_package, period_frame.index)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        strategy_module,
        period_frame,
        params,
        engine_config,
        news_block,
        news_result.enabled,
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
        timeframe="M3",
        schedule_used=schedule_used_from_params(params),
        break_even_setting=params.get("break_even_at_r"),
    )
    return {
        "result": result,
        "summary": news_metrics_from_summary(summary, trades_export),
        "trades_export": trades_export,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
    }


def selection_score(summary: dict[str, Any]) -> float:
    score = 0.0
    score += float(summary["profit_factor"]) * 120.0
    score += float(summary["expectancy_r"]) * 800.0
    score += float(summary["total_return_pct"]) * 1.5
    score -= float(summary["max_drawdown_pct"]) * 3.0
    score -= float(summary["negative_years"]) * 25.0
    score -= float(summary["news_exit_count"]) * 5.0
    if bool(summary.get("insufficient_sample", False)):
        score -= 40.0
    total_trades = int(summary["total_trades"])
    avg_trades_per_month = float(summary["avg_trades_per_month"])
    if total_trades < SERIOUS_DEV_MIN_TRADES:
        score -= 260.0
    elif total_trades < SERIOUS_DEV_MIN_TRADES + 6:
        score -= 90.0
    if avg_trades_per_month < SERIOUS_DEV_MIN_TRADES_PER_MONTH:
        score -= 120.0
    if float(summary["profit_factor"]) <= 1.0:
        score -= 60.0
    if float(summary["expectancy_r"]) <= 0.0:
        score -= 60.0
    return score


def serious_gate_from_row(row: pd.Series) -> bool:
    return (
        int(row["dev_total_trades"]) >= SERIOUS_DEV_MIN_TRADES
        and float(row["dev_avg_trades_per_month"]) >= SERIOUS_DEV_MIN_TRADES_PER_MONTH
        and float(row["dev_profit_factor"]) > SERIOUS_DEV_MIN_PF
        and float(row["dev_expectancy_r"]) > SERIOUS_DEV_MIN_EXPECTANCY
        and int(row["val_total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and int(row["hold_total_trades"]) >= SERIOUS_MIN_VAL_HOLD_TRADES
        and float(row["val_profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
        and float(row["hold_profit_factor"]) >= SERIOUS_MIN_VAL_HOLD_PF
    )


def build_recommendation(selected_row: pd.Series, serious_candidates: int) -> str:
    lines = [
        "# Recomendacion final",
        "",
        f"Estrategia seleccionada para la siguiente fase: **{strategy_module.NAME}**",
        "",
        "Serious gate before verdict:",
        f"- development trades >= {SERIOUS_DEV_MIN_TRADES}",
        f"- development avg trades/month >= {SERIOUS_DEV_MIN_TRADES_PER_MONTH:.2f}",
        f"- development PF > {SERIOUS_DEV_MIN_PF:.2f}",
        f"- development expectancy R > {SERIOUS_DEV_MIN_EXPECTANCY:.2f}",
        f"- validation trades >= {SERIOUS_MIN_VAL_HOLD_TRADES} y PF >= {SERIOUS_MIN_VAL_HOLD_PF:.2f}",
        f"- holdout trades >= {SERIOUS_MIN_VAL_HOLD_TRADES} y PF >= {SERIOUS_MIN_VAL_HOLD_PF:.2f}",
        "",
        "Decision rule:",
        "- Seleccion por score de desarrollo penalizando muestra miserable.",
        "- Validacion y holdout se reportan sin reoptimizar.",
        "",
        "Lectura honesta:",
        f"- development PF: {selected_row['dev_profit_factor']:.3f}",
        f"- validation PF: {selected_row['val_profit_factor']:.3f}",
        f"- holdout PF: {selected_row['hold_profit_factor']:.3f}",
        f"- validation expectancy R: {selected_row['val_expectancy_r']:.4f}",
        f"- holdout expectancy R: {selected_row['hold_expectancy_r']:.4f}",
        "",
        "Veredicto:",
    ]
    if serious_candidates <= 0:
        lines.append("- Ninguna combinacion pasa el serious gate. La linea sigue diagnostica o debe cerrarse segun consistencia final.")
    elif bool(selected_row["serious_candidate"]):
        lines.append("- La linea supera el serious gate y merece seguir bajo auditoria estricta.")
    else:
        lines.append("- La combinacion seleccionada no supera el serious gate. La linea no es defendible todavia.")
    return "\n".join(lines)


def main() -> None:
    build_pm_safe_news_dataset()
    output_root = build_output_root()
    engine_config = build_engine_config()
    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context="pm_micro_reclaim_runner")
    full_frame, full_precision_package = load_m3_precision_context(PAIR, *PERIODS["full_2020_2025"])

    ranking_rows: list[dict[str, Any]] = []
    combos = strategy_module.parameter_grid()
    for combo_id, params in enumerate(combos, start=1):
        dev = evaluate_period(
            frame=full_frame,
            precision_package=full_precision_package,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["development_2020_2023"][0],
            end=PERIODS["development_2020_2023"][1],
        )["summary"]
        val = evaluate_period(
            frame=full_frame,
            precision_package=full_precision_package,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["validation_2024"][0],
            end=PERIODS["validation_2024"][1],
        )["summary"]
        hold = evaluate_period(
            frame=full_frame,
            precision_package=full_precision_package,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=PERIODS["holdout_2025"][0],
            end=PERIODS["holdout_2025"][1],
        )["summary"]
        ranking_rows.append(
            {
                "combo_id": combo_id,
                "selection_score_dev": selection_score(dev),
                "serious_candidate": False,
                "dev_profit_factor": dev["profit_factor"],
                "dev_expectancy_r": dev["expectancy_r"],
                "dev_max_drawdown_pct": dev["max_drawdown_pct"],
                "dev_total_trades": dev["total_trades"],
                "dev_avg_trades_per_month": dev["avg_trades_per_month"],
                "dev_negative_years": dev["negative_years"],
                "dev_news_exit_count": dev["news_exit_count"],
                "val_profit_factor": val["profit_factor"],
                "val_expectancy_r": val["expectancy_r"],
                "val_max_drawdown_pct": val["max_drawdown_pct"],
                "val_total_trades": val["total_trades"],
                "val_avg_trades_per_month": val["avg_trades_per_month"],
                "val_negative_years": val["negative_years"],
                "val_news_exit_count": val["news_exit_count"],
                "hold_profit_factor": hold["profit_factor"],
                "hold_expectancy_r": hold["expectancy_r"],
                "hold_max_drawdown_pct": hold["max_drawdown_pct"],
                "hold_total_trades": hold["total_trades"],
                "hold_avg_trades_per_month": hold["avg_trades_per_month"],
                "hold_negative_years": hold["negative_years"],
                "hold_news_exit_count": hold["news_exit_count"],
                "parameter_set_used": json.dumps(params, ensure_ascii=False),
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["selection_score_dev", "val_profit_factor", "hold_profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking["serious_candidate"] = ranking.apply(serious_gate_from_row, axis=1)
    selected = ranking.iloc[0]
    selected_params = json.loads(str(selected["parameter_set_used"]))
    serious_candidates = int(ranking["serious_candidate"].sum())

    period_payloads: dict[str, dict[str, Any]] = {}
    for period_name, (start, end) in PERIODS.items():
        period_payloads[period_name] = evaluate_period(
            frame=full_frame,
            precision_package=full_precision_package,
            params=selected_params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
            start=start,
            end=end,
        )

    period_overview_rows = []
    for period_name, payload in period_payloads.items():
        summary = payload["summary"]
        period_overview_rows.append(
            {
                "period": period_name,
                "total_trades": summary["total_trades"],
                "profit_factor": summary["profit_factor"],
                "expectancy_r": summary["expectancy_r"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "total_return_pct": summary["total_return_pct"],
                "avg_trades_per_month": summary["avg_trades_per_month"],
                "negative_years": summary["negative_years"],
                "years_positive": summary.get("years_positive", 0),
                "news_exit_count": summary["news_exit_count"],
            }
        )
    period_overview = pd.DataFrame(period_overview_rows)

    selected_dir = output_root / strategy_module.NAME
    export_strategy_bundle(
        selected_dir,
        summary=period_payloads["full_2020_2025"]["summary"],
        trades_export=period_payloads["full_2020_2025"]["trades_export"],
        monthly_stats=period_payloads["full_2020_2025"]["monthly_stats"],
        yearly_stats=period_payloads["full_2020_2025"]["yearly_stats"],
        equity_export=period_payloads["full_2020_2025"]["equity_export"],
        optimization_results=ranking,
        extra_frames={"period_overview.csv": period_overview},
        extra_json={
            "selected_params.json": selected_params,
            "pm_safe_news_summary.json": json.loads(PM_SAFE_NEWS_SUMMARY_FILE.read_text(encoding="utf-8")),
            "selection_metadata.json": {
                "serious_candidates": serious_candidates,
                "selected_combo_id": int(selected["combo_id"]),
                "selected_combo_is_serious_candidate": bool(selected["serious_candidate"]),
                "serious_gate": {
                    "dev_min_trades": SERIOUS_DEV_MIN_TRADES,
                    "dev_min_avg_trades_per_month": SERIOUS_DEV_MIN_TRADES_PER_MONTH,
                    "dev_min_profit_factor": SERIOUS_DEV_MIN_PF,
                    "dev_min_expectancy_r": SERIOUS_DEV_MIN_EXPECTANCY,
                    "val_hold_min_trades_each": SERIOUS_MIN_VAL_HOLD_TRADES,
                    "val_hold_min_profit_factor_each": SERIOUS_MIN_VAL_HOLD_PF,
                },
            },
        },
    )

    ranking.to_csv(output_root / "combo_ranking.csv", index=False)
    period_overview.to_csv(output_root / "period_overview.csv", index=False)
    (output_root / "recomendacion_final.md").write_text(build_recommendation(selected, serious_candidates), encoding="utf-8")

    print(ranking[["combo_id", "selection_score_dev", "dev_profit_factor", "val_profit_factor", "hold_profit_factor"]].to_string(index=False))
    print()
    print(period_overview.to_string(index=False))


if __name__ == "__main__":
    main()
