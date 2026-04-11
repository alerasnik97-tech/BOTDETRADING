#!/usr/bin/env python
"""
Flujo simple y rapido para research FX.

Mantiene:
- data historica ya descargada
- riesgo fijo sobre capital inicial
- proteccion de noticias y shock guard

Recorta:
- walk-forward pesado en cada optimizacion
- familias experimentales complejas en el flujo diario
- reportes innecesarios
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import fx_multi_timeframe_backtester as core


DEFAULT_SIMPLE_PAIRS = ("USDJPY",)
DEFAULT_SCREEN_PAIRS = ("USDJPY", "EURUSD", "USDCAD", "USDCHF")
DEFAULT_SIMPLE_FAMILY = "core_reversion"
ACTIVE_SIMPLE_FAMILIES = (
    DEFAULT_SIMPLE_FAMILY,
    "eurusd_vwap_candidate",
    "eurusd_vwap_frequency_candidate",
)
DEFAULT_SCREEN_FAMILIES = (DEFAULT_SIMPLE_FAMILY,)

FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "core_reversion": {
        "label": "core_session_reversion",
        "base_params": {
            "strategy_family": "core_session_reversion",
            "csr_entry_start": "12:30",
            "csr_entry_end": "15:30",
            "csr_h1_adx_max": 20.0,
            "csr_h1_distance_atr_max": 0.8,
            "csr_h1_ema_spread_atr_max": 0.6,
            "csr_m15_extension_atr": 0.55,
            "csr_m15_rsi_long": 34.0,
            "csr_m15_rsi_short": 66.0,
            "csr_m5_reclaim_rsi_long": 42.0,
            "csr_m5_reclaim_rsi_short": 58.0,
            "csr_reversal_body_atr_min": 0.10,
            "csr_target_atr_buffer": 0.0,
            "csr_take_profit_rr_cap": 0.6,
            "csr_max_hold_bars": 5,
            "context_whitelist_weekdays": "wed_thu,fri",
            "context_whitelist_times": "core",
            "asr_context_min_samples_soft": 100000,
            "asr_context_min_samples_hard": 200000,
        },
        "grids": {
            "ultra_fast": {
                "csr_h1_adx_max": [18.0, 20.0],
                "csr_m15_extension_atr": [0.45, 0.55],
                "stop_atr_multiple": [0.6],
                "csr_take_profit_rr_cap": [0.6],
            },
            "balanced": {
                "csr_h1_adx_max": [16.0, 18.0, 20.0],
                "csr_m15_extension_atr": [0.40, 0.55],
                "csr_reversal_body_atr_min": [0.05, 0.10],
                "stop_atr_multiple": [0.6, 0.8],
                "csr_take_profit_rr_cap": [0.4, 0.6],
                "csr_max_hold_bars": [3, 5],
            },
        },
    },
    "vwap_mean_reversion": {
        "label": "post_expansion_vwap_reversion",
        "base_params": {
            "strategy_family": "post_expansion_mean_reversion",
            "pemr_target_mode": "vwap_only",
            "pemr_entry_start": "11:00",
            "pemr_entry_end": "16:45",
            "pemr_h1_adx_max": 22.0,
            "pemr_h1_distance_atr_max": 1.25,
            "pemr_expansion_atr_min": 1.2,
            "pemr_displacement_atr_min": 0.6,
            "pemr_m5_reclaim_rsi_long": 38.0,
            "pemr_m5_reclaim_rsi_short": 62.0,
            "pemr_m5_trigger_range_atr_max": 1.6,
            "pemr_take_profit_rr_cap": 0.6,
            "pemr_max_hold_bars": 6,
            "pemr_cooldown_minutes": 20,
            "pemr_max_closed_trades_per_day": 2,
            "stop_atr_multiple": 0.7,
        },
        "grids": {
            "ultra_fast": {
                "pemr_h1_adx_max": [18.0, 22.0],
                "pemr_expansion_atr_min": [1.2, 1.5],
                "pemr_displacement_atr_min": [0.6, 0.9],
                "stop_atr_multiple": [0.7],
                "pemr_take_profit_rr_cap": [0.6],
            },
            "balanced": {
                "pemr_h1_adx_max": [18.0, 22.0],
                "pemr_expansion_atr_min": [1.2, 1.5, 1.8],
                "pemr_displacement_atr_min": [0.6, 0.9],
                "stop_atr_multiple": [0.7, 0.9],
                "pemr_take_profit_rr_cap": [0.6, 0.8],
            },
        },
    },
    "eurusd_vwap_candidate": {
        "label": "eurusd_vwap_candidate",
        "base_params": {
            "strategy_family": "post_expansion_mean_reversion",
            "pemr_target_mode": "vwap_only",
            "pemr_entry_start": "11:00",
            "pemr_entry_end": "16:45",
            "pemr_h1_adx_max": 18.0,
            "pemr_h1_distance_atr_max": 1.25,
            "pemr_expansion_atr_min": 1.2,
            "pemr_displacement_atr_min": 0.6,
            "pemr_m5_reclaim_rsi_long": 38.0,
            "pemr_m5_reclaim_rsi_short": 62.0,
            "pemr_m5_trigger_range_atr_max": 1.6,
            "pemr_target_mode": "vwap_only",
            "pemr_take_profit_rr_cap": 0.6,
            "pemr_max_hold_bars": 6,
            "pemr_cooldown_minutes": 20,
            "pemr_max_closed_trades_per_day": 2,
            "context_whitelist_weekdays": "mon_tue,fri",
            "context_whitelist_times": "early",
            "stop_atr_multiple": 0.7,
        },
        "grids": {
            "ultra_fast": {
                "context_whitelist_weekdays": ["mon_tue,fri"],
                "context_whitelist_times": ["early"],
                "stop_atr_multiple": [0.7],
                "pemr_take_profit_rr_cap": [0.6],
            },
            "balanced": {
                "context_whitelist_weekdays": ["mon_tue,fri", "mon_tue"],
                "context_whitelist_times": ["early"],
                "context_whitelist_directions": ["", "short"],
                "stop_atr_multiple": [0.7],
                "pemr_take_profit_rr_cap": [0.6],
            },
        },
    },
    "eurusd_vwap_frequency_candidate": {
        "label": "eurusd_vwap_frequency_candidate",
        "base_params": {
            "strategy_family": "post_expansion_mean_reversion",
            "pemr_target_mode": "vwap_only",
            "pemr_entry_start": "11:00",
            "pemr_entry_end": "16:45",
            "pemr_h1_adx_max": 18.0,
            "pemr_h1_distance_atr_max": 1.25,
            "pemr_expansion_atr_min": 1.2,
            "pemr_displacement_atr_min": 0.6,
            "pemr_m5_reclaim_rsi_long": 38.0,
            "pemr_m5_reclaim_rsi_short": 62.0,
            "pemr_m5_trigger_range_atr_max": 1.6,
            "pemr_take_profit_rr_cap": 0.6,
            "pemr_max_hold_bars": 6,
            "pemr_cooldown_minutes": 20,
            "pemr_max_closed_trades_per_day": 2,
            "context_whitelist_weekdays": "mon_tue,fri",
            "stop_atr_multiple": 0.7,
        },
        "grids": {
            "ultra_fast": {
                "context_whitelist_weekdays": ["mon_tue,fri"],
                "stop_atr_multiple": [0.7],
                "pemr_take_profit_rr_cap": [0.6],
            },
            "balanced": {
                "context_whitelist_weekdays": ["mon_tue,fri", "mon_tue"],
                "context_whitelist_directions": ["", "short"],
                "stop_atr_multiple": [0.7],
                "pemr_take_profit_rr_cap": [0.6],
            },
        },
    },
}


def available_families() -> tuple[str, ...]:
    return tuple(FAMILY_PROFILES.keys())


def build_run_config(
    args: argparse.Namespace,
    *,
    pairs: tuple[str, ...] | None = None,
    start: str | None = None,
    end: str | None = None,
    data_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> core.RunConfig:
    news_file = Path(args.news_file) if args.news_file else None
    pair_values = pairs if pairs is not None else tuple(pair.upper().strip() for pair in args.pairs)
    return core.RunConfig(
        start=start or args.start,
        end=end or args.end,
        pairs=pair_values,
        data_dir=Path(data_dir) if data_dir is not None else Path(args.data_dir),
        report_dir=Path(report_dir) if report_dir is not None else Path(args.report_dir),
        correlation_groups=core.DEFAULT_CORRELATION_GROUPS,
        source=args.source,
        download_missing=args.download_missing,
        force_download=args.force_download,
        strict_data_quality=args.strict_data_quality,
        news_source="csv" if news_file else "none",
        news_file=news_file,
        news_timezone=args.news_timezone,
        news_min_importance=args.news_min_importance,
        hard_news_veto_enabled=not args.disable_hard_news_veto,
        shock_guard_enabled=not args.disable_shock_guard,
        news_no_entry_pre_minutes=args.news_no_entry_pre_minutes,
        news_no_entry_post_minutes=args.news_no_entry_post_minutes,
        news_flatten_minutes_before=args.news_flatten_minutes_before,
        news_hard_no_entry_pre_minutes=args.news_hard_no_entry_pre_minutes,
        news_hard_no_entry_post_minutes=args.news_hard_no_entry_post_minutes,
        news_hard_flatten_minutes_before=args.news_hard_flatten_minutes_before,
        shock_no_entry_atr_multiple=args.shock_no_entry_atr_multiple,
        shock_flatten_atr_multiple=args.shock_flatten_atr_multiple,
        shock_cooldown_minutes=args.shock_cooldown_minutes,
    )


def build_base_params(family: str) -> core.StrategyParameters:
    if family not in FAMILY_PROFILES:
        raise ValueError(f"Familia simple no soportada: {family}")
    return core.StrategyParameters(**FAMILY_PROFILES[family]["base_params"])


def build_broker(args: argparse.Namespace) -> core.BrokerConfig:
    return core.BrokerConfig(
        initial_capital=args.initial_capital,
        risk_fraction=args.risk_pct / 100.0,
        risk_budget_mode="initial_capital",
        commission_rate=args.commission_rate,
        slippage_pips=args.slippage_pips,
        use_spread_model=not args.disable_spread_model,
        max_leverage=args.max_leverage,
        lot_step=args.lot_step,
        session_start="11:00",
        session_end="18:45",
    )


def simple_grid(family: str, preset: str) -> dict[str, list[Any]]:
    if family not in FAMILY_PROFILES:
        raise ValueError(f"Familia simple no soportada: {family}")
    return FAMILY_PROFILES[family]["grids"][preset]


def score_result(result: core.BacktestResult) -> float:
    summary = result.portfolio_summary
    total_trades = int(summary.get("total_trades", 0))
    if total_trades <= 0:
        return -1000.0
    profit_factor = min(float(summary.get("profit_factor", 0.0)), 4.0)
    total_return = float(summary.get("total_return_pct", 0.0))
    max_dd = float(summary.get("max_drawdown_pct", 0.0))
    trades_per_month = float(summary.get("trades_per_month", 0.0))
    win_rate = float(summary.get("win_rate_pct", 0.0))
    score = (
        profit_factor * 40.0
        + total_return * 2.0
        + trades_per_month * 6.0
        + win_rate * 0.6
        - max_dd * 5.0
    )
    if profit_factor < 1.0:
        score -= 30.0
    if total_return < 0.0:
        score -= 20.0
    return score


def export_simple_result(
    result: core.BacktestResult,
    output_dir: Path,
    ranking: pd.DataFrame | None = None,
    walkforward_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    result.monthly_summary.to_csv(output_dir / "monthly_summary.csv", index=False)
    result.yearly_summary.to_csv(output_dir / "yearly_summary.csv", index=False)
    result.news_events.to_csv(output_dir / "news_events.csv", index=False)
    if ranking is not None:
        ranking.to_csv(output_dir / "optimization_ranking.csv", index=False)

    payload = {
        "parameters": core.sanitize_for_json(result.parameters),
        "broker": core.sanitize_for_json(result.broker),
        "portfolio_summary": core.sanitize_for_json(result.portfolio_summary),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    core.export_analysis_bundle(
        output_dir,
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=float(result.broker.get("initial_capital", 0.0)),
        portfolio_summary=result.portfolio_summary,
        parameters=result.parameters,
        best_score=result.robustness_score,
        walkforward_rows=walkforward_rows,
    )


def run_once(run_config: core.RunConfig, params: core.StrategyParameters, broker: core.BrokerConfig) -> core.BacktestResult:
    return core.run_backtest(run_config, params, broker)


def optimize_fast(
    run_config: core.RunConfig,
    base_params: core.StrategyParameters,
    broker: core.BrokerConfig,
    family: str,
    preset: str,
    max_combinations: int,
) -> tuple[core.BacktestResult, pd.DataFrame]:
    grid = simple_grid(family, preset)
    raw_bundle = core.load_raw_bundle(run_config)
    news_events = core.load_news_events(run_config)
    rows: list[dict[str, Any]] = []
    best_result: core.BacktestResult | None = None
    best_score = float("-inf")

    for index, values in enumerate(core.iter_grid(grid), start=1):
        if index > max_combinations:
            break
        candidate = replace(base_params, **values)
        print(f"\n[simple-opt:{family}] Combinacion {index}: {values}")
        result = core.run_backtest(run_config, candidate, broker, raw_bundle=raw_bundle, news_events=news_events)
        score = score_result(result)
        rows.append(
            {
                "family": family,
                **values,
                "score": score,
                **result.portfolio_summary,
            }
        )
        if best_result is None or score > best_score:
            best_result = result
            best_score = score

    if best_result is None:
        raise RuntimeError("No se pudo evaluar ninguna combinacion.")

    ranking = pd.DataFrame(rows).sort_values(
        by=["score", "profit_factor", "total_return_pct", "trades_per_month"],
        ascending=[False, False, False, False],
    )
    best_result.robustness_score = best_score
    return best_result, ranking.reset_index(drop=True)


def summarize_result_row(family: str, pair: str, stage: str, result: core.BacktestResult) -> dict[str, Any]:
    summary = result.portfolio_summary
    return {
        "family": family,
        "pair": pair,
        "stage": stage,
        "score": score_result(result),
        "total_return_pct": summary["total_return_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "profit_factor": summary["profit_factor"],
        "total_trades": summary["total_trades"],
        "trades_per_month": summary["trades_per_month"],
        "win_rate_pct": summary["win_rate_pct"],
    }


def build_walkforward_row(
    *,
    train_period: str,
    test_period: str,
    params: dict[str, Any],
    result: core.BacktestResult,
) -> dict[str, Any]:
    return {
        "train_period": train_period,
        "test_period": test_period,
        "params_used": json.dumps(core.sanitize_for_json(params), ensure_ascii=True),
        "trades": int(result.portfolio_summary.get("total_trades", 0)),
        "win_rate": float(result.portfolio_summary.get("win_rate_pct", 0.0)),
        "pnl_r": float(result.trades["r_multiple"].sum()) if not result.trades.empty and "r_multiple" in result.trades.columns else 0.0,
        "pnl_usd": float(result.trades["net_pnl_usd"].sum()) if not result.trades.empty and "net_pnl_usd" in result.trades.columns else 0.0,
        "max_drawdown_pct": float(result.portfolio_summary.get("max_drawdown_pct", 0.0)),
        "profit_factor": float(result.portfolio_summary.get("profit_factor", 0.0)),
    }


def build_screen_ranking(summary_rows: pd.DataFrame) -> pd.DataFrame:
    design = summary_rows[summary_rows["stage"] == "design"].copy()
    oos = summary_rows[summary_rows["stage"] == "oos"].copy()
    design = design.add_prefix("design_").rename(columns={"design_family": "family", "design_pair": "pair"})
    oos = oos.add_prefix("oos_").rename(columns={"oos_family": "family", "oos_pair": "pair"})
    ranking = design.merge(oos, on=["family", "pair"], how="outer")
    ranking["accepted"] = (
        (ranking["design_total_return_pct"] >= 0.0)
        & (ranking["design_profit_factor"] >= 1.0)
        & (ranking["oos_total_return_pct"] >= 0.0)
        & (ranking["oos_profit_factor"] >= 1.0)
        & (ranking["oos_trades_per_month"] >= 0.5)
    )
    ranking["screen_score"] = (
        ranking["accepted"].astype(int) * 150
        + ranking["oos_score"] * 0.7
        + ranking["design_score"] * 0.3
        + ranking["oos_trades_per_month"] * 12
        - ranking["oos_max_drawdown_pct"] * 4
    )
    ranking = ranking.sort_values(
        by=["screen_score", "oos_total_return_pct", "oos_profit_factor"],
        ascending=[False, False, False],
    )
    return ranking.reset_index(drop=True)


def build_screen_decision(ranking: pd.DataFrame) -> dict[str, Any]:
    accepted = ranking.loc[ranking["accepted"]].copy()
    if accepted.empty:
        top = ranking.iloc[0].to_dict() if not ranking.empty else {}
        return {
            "decision": "no_survivor",
            "survivors": [],
            "recommended_family": top.get("family"),
            "recommended_pair": top.get("pair"),
        }
    best = accepted.iloc[0]
    family_summary = (
        accepted.groupby("family", as_index=False)
        .agg(
            survivor_pairs=("pair", "count"),
            avg_oos_return_pct=("oos_total_return_pct", "mean"),
            avg_oos_profit_factor=("oos_profit_factor", "mean"),
            avg_oos_trades_per_month=("oos_trades_per_month", "mean"),
        )
        .sort_values(["survivor_pairs", "avg_oos_return_pct", "avg_oos_profit_factor"], ascending=[False, False, False])
    )
    return {
        "decision": "survivor_found",
        "survivors": accepted[["family", "pair"]].to_dict(orient="records"),
        "recommended_family": best["family"],
        "recommended_pair": best["pair"],
        "family_summary": core.sanitize_for_json(family_summary.to_dict(orient="records")),
    }


def run_screen(args: argparse.Namespace, broker: core.BrokerConfig) -> None:
    output_root = core.build_report_dir(Path(args.report_dir), "simple_screen")
    rows: list[dict[str, Any]] = []

    for family in args.families:
        base_params = build_base_params(family)
        for pair in args.pairs:
            pair = pair.upper().strip()
            print(f"\n[simple-screen] Familia={family} Pair={pair}")
            design_config = build_run_config(
                args,
                pairs=(pair,),
                start=args.design_start,
                end=args.design_end,
                data_dir=args.design_data_dir,
                report_dir=output_root / family / pair / "design",
            )
            best_result, ranking = optimize_fast(
                run_config=design_config,
                base_params=base_params,
                broker=broker,
                family=family,
                preset=args.preset,
                max_combinations=args.max_combinations,
            )
            design_dir = output_root / family / pair / "design"
            design_walkforward = [
                build_walkforward_row(
                    train_period=f"{args.design_start}:{args.design_end}",
                    test_period="",
                    params=best_result.parameters,
                    result=best_result,
                )
            ]
            export_simple_result(best_result, design_dir, ranking, walkforward_rows=design_walkforward)
            rows.append(summarize_result_row(family, pair, "design", best_result))

            oos_config = build_run_config(
                args,
                pairs=(pair,),
                start=args.oos_start,
                end=args.oos_end,
                data_dir=args.oos_data_dir,
                report_dir=output_root / family / pair / "oos",
            )
            oos_result = run_once(oos_config, core.StrategyParameters(**best_result.parameters), broker)
            oos_dir = output_root / family / pair / "oos"
            oos_walkforward = [
                build_walkforward_row(
                    train_period=f"{args.design_start}:{args.design_end}",
                    test_period=f"{args.oos_start}:{args.oos_end}",
                    params=best_result.parameters,
                    result=oos_result,
                )
            ]
            export_simple_result(oos_result, oos_dir, walkforward_rows=oos_walkforward)
            rows.append(summarize_result_row(family, pair, "oos", oos_result))

    summary = pd.DataFrame(rows)
    ranking = build_screen_ranking(summary)
    decision = build_screen_decision(ranking)
    summary.to_csv(output_root / "screen_summary.csv", index=False)
    ranking.to_csv(output_root / "screen_ranking.csv", index=False)
    (output_root / "screen_decision.json").write_text(json.dumps(core.sanitize_for_json(decision), indent=2), encoding="utf-8")

    print("\n=== SIMPLE SCREEN SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SIMPLE SCREEN RANKING ===")
    print(ranking.to_string(index=False))
    print("\n=== SIMPLE SCREEN DECISION ===")
    print(json.dumps(core.sanitize_for_json(decision), indent=2))
    print(f"\nReportes exportados en: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot simple y rapido para research FX.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--pairs", nargs="+", default=list(DEFAULT_SIMPLE_PAIRS))
        target.add_argument("--data-dir", default="data_free_2020/prepared")
        target.add_argument("--report-dir", default="reports_simple_bot")
        target.add_argument("--source", choices=["local", "auto", "dukascopy"], default="local")
        target.add_argument("--download-missing", action="store_true")
        target.add_argument("--force-download", action="store_true")
        target.add_argument("--strict-data-quality", action="store_true")
        target.add_argument("--news-file", default="data/forex_factory_cache.csv")
        target.add_argument("--news-timezone", default="UTC")
        target.add_argument("--news-min-importance", type=int, default=3)
        target.add_argument("--disable-hard-news-veto", action="store_true")
        target.add_argument("--disable-shock-guard", action="store_true")
        target.add_argument("--news-no-entry-pre-minutes", type=int, default=45)
        target.add_argument("--news-no-entry-post-minutes", type=int, default=30)
        target.add_argument("--news-flatten-minutes-before", type=int, default=15)
        target.add_argument("--news-hard-no-entry-pre-minutes", type=int, default=90)
        target.add_argument("--news-hard-no-entry-post-minutes", type=int, default=60)
        target.add_argument("--news-hard-flatten-minutes-before", type=int, default=30)
        target.add_argument("--shock-no-entry-atr-multiple", type=float, default=2.5)
        target.add_argument("--shock-flatten-atr-multiple", type=float, default=3.0)
        target.add_argument("--shock-cooldown-minutes", type=int, default=30)
        target.add_argument("--initial-capital", type=float, default=100000.0)
        target.add_argument("--risk-pct", type=float, default=1.0)
        target.add_argument("--commission-rate", type=float, default=0.00002)
        target.add_argument("--slippage-pips", type=float, default=0.2)
        target.add_argument("--disable-spread-model", action="store_true")
        target.add_argument("--max-leverage", type=float, default=20.0)
        target.add_argument("--lot-step", type=int, default=1000)

    run_parser = subparsers.add_parser("run", help="Ejecuta una sola corrida simple.")
    add_common_flags(run_parser)
    run_parser.add_argument("--start", default="2020-01-01")
    run_parser.add_argument("--end", default="2021-12-31")
    run_parser.add_argument("--family", choices=ACTIVE_SIMPLE_FAMILIES, default=DEFAULT_SIMPLE_FAMILY)

    opt_parser = subparsers.add_parser("optimize", help="Optimiza rapido sin walk-forward pesado.")
    add_common_flags(opt_parser)
    opt_parser.add_argument("--start", default="2020-01-01")
    opt_parser.add_argument("--end", default="2021-12-31")
    opt_parser.add_argument("--family", choices=ACTIVE_SIMPLE_FAMILIES, default=DEFAULT_SIMPLE_FAMILY)
    opt_parser.add_argument("--preset", choices=["ultra_fast", "balanced"], default="ultra_fast")
    opt_parser.add_argument("--max-combinations", type=int, default=16)

    screen_parser = subparsers.add_parser("screen", help="Screen rapido de familias simples sobre diseno y OOS.")
    add_common_flags(screen_parser)
    screen_parser.set_defaults(pairs=list(DEFAULT_SCREEN_PAIRS))
    screen_parser.add_argument("--families", nargs="+", choices=available_families(), default=list(DEFAULT_SCREEN_FAMILIES))
    screen_parser.add_argument("--design-start", default="2020-01-01")
    screen_parser.add_argument("--design-end", default="2021-12-31")
    screen_parser.add_argument("--design-data-dir", default="data_free_2020/prepared")
    screen_parser.add_argument("--oos-start", default="2022-01-01")
    screen_parser.add_argument("--oos-end", default="2025-12-31")
    screen_parser.add_argument("--oos-data-dir", default="data_candidates_2022_2025/prepared")
    screen_parser.add_argument("--preset", choices=["ultra_fast", "balanced"], default="ultra_fast")
    screen_parser.add_argument("--max-combinations", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    broker = build_broker(args)

    if args.command == "screen":
        run_screen(args, broker)
        return

    run_config = build_run_config(args)
    params = build_base_params(args.family)

    if args.command == "run":
        result = run_once(run_config, params, broker)
        report_dir = core.build_report_dir(Path(args.report_dir), "simple_run")
        walkforward_rows = [
            build_walkforward_row(
                train_period="",
                test_period=f"{args.start}:{args.end}",
                params=result.parameters,
                result=result,
            )
        ]
        export_simple_result(result, report_dir, walkforward_rows=walkforward_rows)
        core.print_run_summary(result)
        print(f"\nReportes exportados en: {report_dir}")
        return

    best_result, ranking = optimize_fast(
        run_config=run_config,
        base_params=params,
        broker=broker,
        family=args.family,
        preset=args.preset,
        max_combinations=args.max_combinations,
    )
    report_dir = core.build_report_dir(Path(args.report_dir), "simple_optimize")
    walkforward_rows = [
        build_walkforward_row(
            train_period=f"{args.start}:{args.end}",
            test_period="",
            params=best_result.parameters,
            result=best_result,
        )
    ]
    export_simple_result(best_result, report_dir, ranking, walkforward_rows=walkforward_rows)
    print("\n=== TOP PARAMETROS ===")
    print(ranking.head(10).to_string(index=False))
    print("\n=== MEJOR CONFIGURACION ===")
    core.print_run_summary(best_result)
    print(f"\nReportes exportados en: {report_dir}")


if __name__ == "__main__":
    main()
