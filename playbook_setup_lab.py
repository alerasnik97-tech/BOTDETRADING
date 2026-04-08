#!/usr/bin/env python
"""
Laboratorio de setups para usdjp_session_playbook.

Optimiza cada setup por separado en una ventana de diseno y, opcionalmente,
corre la mejor configuracion en una ventana fuera de muestra.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import fx_multi_timeframe_backtester as bt


SETUP_FLAGS = {
    "core_reversion": {
        "pb_enable_core_reversion": True,
        "pb_enable_late_continuation": False,
        "pb_enable_compression_breakout": False,
    },
    "late_continuation": {
        "pb_enable_core_reversion": False,
        "pb_enable_late_continuation": True,
        "pb_enable_compression_breakout": False,
    },
    "compression_breakout": {
        "pb_enable_core_reversion": False,
        "pb_enable_late_continuation": False,
        "pb_enable_compression_breakout": True,
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


def summarize_row(setup_name: str, stage: str, result: bt.BacktestResult) -> dict[str, float | str]:
    summary = result.portfolio_summary
    return {
        "setup_name": setup_name,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorio de setups del playbook USDJPY.")
    parser.add_argument("--pair", default="USDJPY")
    parser.add_argument("--design-start", default="2020-01-01")
    parser.add_argument("--design-end", default="2021-12-31")
    parser.add_argument("--design-data-dir", default="data_free_2020/prepared")
    parser.add_argument("--oos-start", default=None)
    parser.add_argument("--oos-end", default=None)
    parser.add_argument("--oos-data-dir", default=None)
    parser.add_argument("--source", choices=["local", "auto", "dukascopy"], default="local")
    parser.add_argument("--news-file", required=True)
    parser.add_argument("--news-timezone", default="UTC")
    parser.add_argument("--news-min-importance", type=int, default=3)
    parser.add_argument("--disable-hard-news-veto", action="store_true")
    parser.add_argument("--disable-shock-guard", action="store_true")
    parser.add_argument("--strict-data-quality", action="store_true")
    parser.add_argument("--optimization-profile", choices=["consistency", "winrate", "frequency"], default="consistency")
    parser.add_argument("--max-combinations", type=int, default=24)
    parser.add_argument("--report-dir", default="reports_playbook_setup_lab")
    parser.add_argument("--setup-names", nargs="+", choices=sorted(SETUP_FLAGS), default=sorted(SETUP_FLAGS))
    args = parser.parse_args()

    broker = bt.BrokerConfig()
    design_config = build_run_config(args, args.design_start, args.design_end, args.design_data_dir)
    oos_config = None
    if args.oos_start and args.oos_end and args.oos_data_dir:
        oos_config = build_run_config(args, args.oos_start, args.oos_end, args.oos_data_dir)

    rows: list[dict[str, float | str]] = []
    output_root = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for setup_name in args.setup_names:
        flags = SETUP_FLAGS[setup_name]
        print(f"\n[lab] Optimizando setup: {setup_name}")
        base_params = bt.StrategyParameters(strategy_family="usdjp_session_playbook", **flags)
        best_result, ranking = bt.run_grid_search(
            run_config=design_config,
            base_params=base_params,
            broker=broker,
            optimization_profile=args.optimization_profile,
            max_combinations=args.max_combinations,
        )

        setup_dir = output_root / setup_name
        design_dir = setup_dir / "design"
        bt.export_result(best_result, design_dir)
        ranking.to_csv(design_dir / "optimization_ranking.csv", index=False)
        rows.append(summarize_row(setup_name, "design", best_result))

        if oos_config is not None:
            oos_params = bt.StrategyParameters(**best_result.parameters)
            oos_news = bt.load_news_events(oos_config)
            oos_result = bt.run_backtest(oos_config, oos_params, broker, news_events=oos_news)
            oos_dir = setup_dir / "oos"
            bt.export_result(oos_result, oos_dir)
            rows.append(summarize_row(setup_name, "oos", oos_result))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "setup_lab_summary.csv", index=False)
    (output_root / "setup_lab_summary.json").write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print("\n=== RESUMEN LABORATORIO ===")
    print(comparison.to_string(index=False))
    print(f"\nLaboratorio exportado en: {output_root}")


if __name__ == "__main__":
    main()
