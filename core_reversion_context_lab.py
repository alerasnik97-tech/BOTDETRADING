#!/usr/bin/env python
"""
Laboratorio de filtros de contexto para el setup core_reversion.

Toma la mejor configuracion base encontrada para core_reversion y prueba
whitelists de contexto pequenos y explicitamente definidos. La idea es
subir robustez por seleccion de contexto, no por aflojar la entrada.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import fx_multi_timeframe_backtester as bt


CONTEXT_CANDIDATES: dict[str, dict[str, str]] = {
    "base": {},
    "core_only": {
        "context_whitelist_times": "core",
    },
    "core_balanced_or_calm": {
        "context_whitelist_times": "core",
        "context_whitelist_regimes": "balanced,calm",
    },
    "wed_thu_fri_core": {
        "context_whitelist_weekdays": "wed_thu,fri",
        "context_whitelist_times": "core",
    },
    "wed_thu_fri_core_balanced_or_calm": {
        "context_whitelist_weekdays": "wed_thu,fri",
        "context_whitelist_times": "core",
        "context_whitelist_regimes": "balanced,calm",
    },
    "short_only": {
        "context_whitelist_directions": "short",
    },
    "core_short_only": {
        "context_whitelist_times": "core",
        "context_whitelist_directions": "short",
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
        raise ValueError("El laboratorio de contexto espera una configuracion base usdjp_session_playbook.")
    if not params.pb_enable_core_reversion or params.pb_enable_late_continuation or params.pb_enable_compression_breakout:
        raise ValueError("La configuracion base debe tener solo core_reversion habilitado.")
    return params


def summarize_row(candidate_name: str, stage: str, filters: dict[str, str], result: bt.BacktestResult) -> dict[str, float | str]:
    summary = result.portfolio_summary
    return {
        "candidate_name": candidate_name,
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
        "whitelist_weekdays": filters.get("context_whitelist_weekdays", ""),
        "whitelist_times": filters.get("context_whitelist_times", ""),
        "whitelist_regimes": filters.get("context_whitelist_regimes", ""),
        "whitelist_extensions": filters.get("context_whitelist_extensions", ""),
        "whitelist_directions": filters.get("context_whitelist_directions", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorio de filtros de contexto para core_reversion.")
    parser.add_argument("--pair", default="USDJPY")
    parser.add_argument("--base-summary", default="reports_playbook_setup_lab/core_reversion/design/summary.json")
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
    parser.add_argument("--candidate-names", nargs="+", choices=sorted(CONTEXT_CANDIDATES), default=sorted(CONTEXT_CANDIDATES))
    parser.add_argument("--report-dir", default="reports_core_reversion_context_lab")
    args = parser.parse_args()

    base_params = load_base_params(args.base_summary)
    broker = bt.BrokerConfig()
    design_config = build_run_config(args, args.design_start, args.design_end, args.design_data_dir)
    design_news = bt.load_news_events(design_config)

    oos_config = None
    oos_news = None
    if args.oos_start and args.oos_end and args.oos_data_dir:
        oos_config = build_run_config(args, args.oos_start, args.oos_end, args.oos_data_dir)
        oos_news = bt.load_news_events(oos_config)

    output_root = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for candidate_name in args.candidate_names:
        filters = CONTEXT_CANDIDATES[candidate_name]
        params = replace(base_params, **filters)
        print(f"\n[context-lab] Evaluando candidato: {candidate_name}")

        design_result = bt.run_backtest(design_config, params, broker, news_events=design_news)
        design_dir = output_root / candidate_name / "design"
        bt.export_result(design_result, design_dir)
        rows.append(summarize_row(candidate_name, "design", filters, design_result))

        if oos_config is not None and oos_news is not None:
            oos_result = bt.run_backtest(oos_config, params, broker, news_events=oos_news)
            oos_dir = output_root / candidate_name / "oos"
            bt.export_result(oos_result, oos_dir)
            rows.append(summarize_row(candidate_name, "oos", filters, oos_result))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "context_lab_summary.csv", index=False)
    (output_root / "context_lab_summary.json").write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print("\n=== RESUMEN CONTEXT LAB ===")
    print(comparison.to_string(index=False))
    print(f"\nLaboratorio exportado en: {output_root}")


if __name__ == "__main__":
    main()
