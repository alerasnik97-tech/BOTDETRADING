#!/usr/bin/env python
"""
Stress test del setup superviviente USDJPY core_reversion.

Compara la misma configuracion bajo escenarios de ejecucion mas duros
para validar si el edge sobrevive antes de intentar escalar frecuencia
o agregar nuevos pares.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import fx_multi_timeframe_backtester as bt


SCENARIOS: dict[str, dict[str, object]] = {
    "baseline_no_shock": {
        "broker": {},
        "run": {
            "shock_guard_enabled": False,
        },
    },
    "production_layered": {
        "broker": {},
        "run": {
            "shock_guard_enabled": True,
        },
    },
    "high_costs_layered": {
        "broker": {
            "commission_rate": 0.00003,
            "slippage_pips": 0.35,
        },
        "run": {
            "shock_guard_enabled": True,
        },
    },
    "extreme_costs_layered": {
        "broker": {
            "commission_rate": 0.00004,
            "slippage_pips": 0.50,
        },
        "run": {
            "shock_guard_enabled": True,
        },
    },
}


def build_run_config(args: argparse.Namespace, start: str, end: str, data_dir: str) -> bt.RunConfig:
    return bt.RunConfig(
        start=start,
        end=end,
        pairs=(args.pair,),
        data_dir=Path(data_dir),
        report_dir=Path(args.report_dir),
        source=args.source,
        strict_data_quality=args.strict_data_quality,
        news_source="csv",
        news_file=Path(args.news_file),
        news_timezone=args.news_timezone,
        news_min_importance=args.news_min_importance,
        hard_news_veto_enabled=not args.disable_hard_news_veto,
        shock_guard_enabled=not args.disable_shock_guard,
    )


def load_base_params(summary_path: str) -> bt.StrategyParameters:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    params = bt.StrategyParameters(**payload["parameters"])
    if params.strategy_family != "usdjp_session_playbook":
        raise ValueError("El stress test espera una configuracion base usdjp_session_playbook.")
    return params


def summarize_row(
    scenario_name: str,
    stage: str,
    broker: bt.BrokerConfig,
    run_config: bt.RunConfig,
    result: bt.BacktestResult,
) -> dict[str, float | str]:
    summary = result.portfolio_summary
    return {
        "scenario_name": scenario_name,
        "stage": stage,
        "final_equity": summary["final_equity"],
        "total_return_pct": summary["total_return_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "sharpe_ratio": summary["sharpe_ratio"],
        "profit_factor": summary["profit_factor"],
        "total_trades": summary["total_trades"],
        "wins": summary["wins"],
        "losses": summary["losses"],
        "win_rate_pct": summary["win_rate_pct"],
        "trades_per_month": summary["trades_per_month"],
        "robustness_score": result.robustness_score,
        "commission_rate": broker.commission_rate,
        "slippage_pips": broker.slippage_pips,
        "spread_model": broker.use_spread_model,
        "shock_guard_enabled": run_config.shock_guard_enabled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test del setup superviviente USDJPY core_reversion.")
    parser.add_argument("--pair", default="USDJPY")
    parser.add_argument("--base-summary", default="reports_playbook_setup_lab/core_reversion/design/summary.json")
    parser.add_argument("--design-start", default="2016-01-01")
    parser.add_argument("--design-end", default="2021-12-31")
    parser.add_argument("--design-data-dir", default="data_usdjpy_2016_2021/prepared")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--oos-data-dir", default="data_usdjpy_2022_2025/prepared")
    parser.add_argument("--source", choices=["local", "auto", "dukascopy"], default="local")
    parser.add_argument("--news-file", required=True)
    parser.add_argument("--news-timezone", default="UTC")
    parser.add_argument("--news-min-importance", type=int, default=3)
    parser.add_argument("--disable-hard-news-veto", action="store_true")
    parser.add_argument("--disable-shock-guard", action="store_true")
    parser.add_argument("--strict-data-quality", action="store_true")
    parser.add_argument("--report-dir", default="reports_survivor_stress_test")
    parser.add_argument("--scenario-names", nargs="+", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    args = parser.parse_args()

    base_params = load_base_params(args.base_summary)
    params = replace(
        base_params,
        context_whitelist_weekdays="wed_thu,fri",
        context_whitelist_times="core",
    )

    base_broker = bt.BrokerConfig()
    design_config = build_run_config(args, args.design_start, args.design_end, args.design_data_dir)
    oos_config = build_run_config(args, args.oos_start, args.oos_end, args.oos_data_dir)

    output_root = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for scenario_name in args.scenario_names:
        scenario = SCENARIOS[scenario_name]
        broker = replace(base_broker, **scenario["broker"])
        design_run = replace(design_config, **scenario["run"])
        oos_run = replace(oos_config, **scenario["run"])

        print(f"\n[stress-test] Evaluando escenario: {scenario_name}")

        design_news = bt.load_news_events(design_run)
        design_result = bt.run_backtest(design_run, params, broker, news_events=design_news)
        design_dir = output_root / scenario_name / "design"
        bt.export_result(design_result, design_dir)
        rows.append(summarize_row(scenario_name, "design", broker, design_run, design_result))

        oos_news = bt.load_news_events(oos_run)
        oos_result = bt.run_backtest(oos_run, params, broker, news_events=oos_news)
        oos_dir = output_root / scenario_name / "oos"
        bt.export_result(oos_result, oos_dir)
        rows.append(summarize_row(scenario_name, "oos", broker, oos_run, oos_result))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "stress_test_summary.csv", index=False)
    (output_root / "stress_test_summary.json").write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print("\n=== RESUMEN STRESS TEST ===")
    print(comparison.to_string(index=False))
    print(f"\nStress test exportado en: {output_root}")


if __name__ == "__main__":
    main()
