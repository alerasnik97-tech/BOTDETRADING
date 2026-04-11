#!/usr/bin/env python
"""
Compara el impacto de las guardas de noticias sobre una configuracion fija.

Escenarios:
1. unprotected: sin calendario ni shock guard
2. calendar_only: calendario economico + hard veto por evento
3. layered_guard: calendario economico + hard veto + shock guard
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

import fx_multi_timeframe_backtester as bt


def load_summary(summary_path: Path) -> tuple[bt.StrategyParameters, bt.BrokerConfig]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    params = bt.StrategyParameters(**payload["parameters"])
    broker = bt.BrokerConfig(**payload["broker"])
    return params, broker


def build_run_config(args: argparse.Namespace) -> bt.RunConfig:
    return bt.RunConfig(
        start=args.start,
        end=args.end,
        pairs=(args.pair,),
        data_dir=Path(args.data_dir),
        report_dir=Path(args.report_dir),
        source=args.source,
        strict_data_quality=args.strict_data_quality,
        news_source="none",
        news_file=None,
        news_timezone=args.news_timezone,
    )


def summarize_result(name: str, result: bt.BacktestResult) -> dict[str, float | str]:
    summary = dict(result.portfolio_summary)
    return {
        "scenario": name,
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
        "news_events_count": len(result.news_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara guardas de noticias sobre una configuracion fija.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--pair", default="USDJPY")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--source", choices=["local", "auto", "dukascopy"], default="local")
    parser.add_argument("--data-dir", default="data_free_2020/prepared")
    parser.add_argument("--report-dir", default="reports_compare_news_guard")
    parser.add_argument("--news-file", required=True)
    parser.add_argument("--news-timezone", default="UTC")
    parser.add_argument("--strict-data-quality", action="store_true")
    args = parser.parse_args()

    params, broker = load_summary(Path(args.summary_json))
    base_run_config = build_run_config(args)
    raw_bundle = bt.load_raw_bundle(base_run_config)

    scenarios: list[tuple[str, bt.RunConfig]] = [
        (
            "unprotected",
            replace(
                base_run_config,
                news_source="none",
                news_file=None,
                hard_news_veto_enabled=False,
                shock_guard_enabled=False,
            ),
        ),
        (
            "calendar_only",
            replace(
                base_run_config,
                news_source="csv",
                news_file=Path(args.news_file),
                hard_news_veto_enabled=True,
                shock_guard_enabled=False,
            ),
        ),
        (
            "layered_guard",
            replace(
                base_run_config,
                news_source="csv",
                news_file=Path(args.news_file),
                hard_news_veto_enabled=True,
                shock_guard_enabled=True,
            ),
        ),
    ]

    output_root = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    comparison_payload: dict[str, dict[str, float | str]] = {}

    for scenario_name, scenario_config in scenarios:
        news_events = bt.load_news_events(scenario_config)
        result = bt.run_backtest(
            run_config=scenario_config,
            params=params,
            broker=broker,
            raw_bundle=raw_bundle,
            news_events=news_events,
        )
        scenario_dir = output_root / scenario_name
        bt.export_result(result, scenario_dir)
        row = summarize_result(scenario_name, result)
        rows.append(row)
        comparison_payload[scenario_name] = row

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "comparison.csv", index=False)
    (output_root / "comparison.json").write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"\nComparacion exportada en: {output_root}")


if __name__ == "__main__":
    main()
