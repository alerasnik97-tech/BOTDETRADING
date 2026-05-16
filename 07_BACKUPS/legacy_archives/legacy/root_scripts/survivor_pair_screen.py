#!/usr/bin/env python
"""
Screen de pares candidatos para la linea superviviente.

Toma la configuracion superviviente de USDJPY y la transpone a otros pares
sin reoptimizarla. El objetivo es encontrar bloques de cartera que mantengan
calidad bajo el mismo marco de ejecucion antes de intentar subir frecuencia.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import fx_multi_timeframe_backtester as bt


def build_run_config(
    pair: str,
    start: str,
    end: str,
    data_dir: str,
    report_dir: str,
    source: str,
    strict_data_quality: bool,
    news_file: str,
    news_timezone: str,
    news_min_importance: int,
    disable_hard_news_veto: bool,
) -> bt.RunConfig:
    return bt.RunConfig(
        start=start,
        end=end,
        pairs=(pair,),
        data_dir=Path(data_dir),
        report_dir=Path(report_dir),
        source=source,
        strict_data_quality=strict_data_quality,
        news_source="csv",
        news_file=Path(news_file),
        news_timezone=news_timezone,
        news_min_importance=news_min_importance,
        hard_news_veto_enabled=not disable_hard_news_veto,
        shock_guard_enabled=True,
    )


def load_base_params(summary_path: str) -> bt.StrategyParameters:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    params = bt.StrategyParameters(**payload["parameters"])
    if params.strategy_family != "usdjp_session_playbook":
        raise ValueError("El screen espera una configuracion base usdjp_session_playbook.")
    return replace(
        params,
        context_whitelist_weekdays="wed_thu,fri",
        context_whitelist_times="core",
    )


def summarize_row(pair: str, stage: str, result: bt.BacktestResult) -> dict[str, float | str]:
    summary = result.portfolio_summary
    return {
        "pair": pair,
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


def build_pair_ranking(summary_rows: pd.DataFrame) -> pd.DataFrame:
    design = summary_rows[summary_rows["stage"] == "design"].copy()
    oos = summary_rows[summary_rows["stage"] == "oos"].copy()
    design = design.add_prefix("design_").rename(columns={"design_pair": "pair"})
    oos = oos.add_prefix("oos_").rename(columns={"oos_pair": "pair"})
    ranking = design.merge(oos, on="pair", how="outer")
    ranking["design_pass"] = (
        (ranking["design_total_return_pct"] >= 0.0)
        & (ranking["design_profit_factor"] >= 1.0)
        & (ranking["design_max_drawdown_pct"] <= 3.0)
    )
    ranking["oos_pass"] = (
        (ranking["oos_total_return_pct"] >= 0.0)
        & (ranking["oos_profit_factor"] >= 1.0)
        & (ranking["oos_max_drawdown_pct"] <= 3.0)
    )
    ranking["promotion_score"] = (
        ranking["oos_pass"].astype(int) * 100
        + ranking["design_pass"].astype(int) * 40
        + ranking["oos_profit_factor"].replace(float("inf"), 5.0) * 10
        + ranking["design_profit_factor"].replace(float("inf"), 5.0) * 5
        - ranking["oos_max_drawdown_pct"] * 2
        - ranking["design_max_drawdown_pct"]
    )
    ranking = ranking.sort_values(
        by=[
            "promotion_score",
            "oos_profit_factor",
            "oos_total_return_pct",
            "design_profit_factor",
            "design_total_return_pct",
        ],
        ascending=[False, False, False, False, False],
    )
    return ranking.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen de pares para la linea superviviente.")
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--base-summary", default="reports_playbook_setup_lab/core_reversion/design/summary.json")
    parser.add_argument("--design-start", default="2020-01-01")
    parser.add_argument("--design-end", default="2021-12-31")
    parser.add_argument("--design-data-dir", default="data_free_2020/prepared")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--oos-data-dir", default="data_candidates_2022_2025/prepared")
    parser.add_argument("--source", choices=["local", "auto", "dukascopy"], default="local")
    parser.add_argument("--news-file", required=True)
    parser.add_argument("--news-timezone", default="UTC")
    parser.add_argument("--news-min-importance", type=int, default=3)
    parser.add_argument("--disable-hard-news-veto", action="store_true")
    parser.add_argument("--strict-data-quality", action="store_true")
    parser.add_argument("--report-dir", default="reports_survivor_pair_screen")
    args = parser.parse_args()

    params = load_base_params(args.base_summary)
    broker = bt.BrokerConfig()
    output_root = Path(args.report_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []

    for pair in args.pairs:
        pair = pair.upper().strip()
        print(f"\n[pair-screen] Evaluando par: {pair}")

        design_run = build_run_config(
            pair,
            args.design_start,
            args.design_end,
            args.design_data_dir,
            args.report_dir,
            args.source,
            args.strict_data_quality,
            args.news_file,
            args.news_timezone,
            args.news_min_importance,
            args.disable_hard_news_veto,
        )
        design_news = bt.load_news_events(design_run)
        design_result = bt.run_backtest(design_run, params, broker, news_events=design_news)
        design_dir = output_root / pair / "design"
        bt.export_result(design_result, design_dir)
        rows.append(summarize_row(pair, "design", design_result))

        oos_run = build_run_config(
            pair,
            args.oos_start,
            args.oos_end,
            args.oos_data_dir,
            args.report_dir,
            args.source,
            args.strict_data_quality,
            args.news_file,
            args.news_timezone,
            args.news_min_importance,
            args.disable_hard_news_veto,
        )
        oos_news = bt.load_news_events(oos_run)
        oos_result = bt.run_backtest(oos_run, params, broker, news_events=oos_news)
        oos_dir = output_root / pair / "oos"
        bt.export_result(oos_result, oos_dir)
        rows.append(summarize_row(pair, "oos", oos_result))

    summary = pd.DataFrame(rows)
    ranking = build_pair_ranking(summary)
    summary.to_csv(output_root / "pair_screen_summary.csv", index=False)
    ranking.to_csv(output_root / "pair_screen_ranking.csv", index=False)
    (output_root / "pair_screen_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print("\n=== RESUMEN PAIR SCREEN ===")
    print(summary.to_string(index=False))
    print("\n=== RANKING ===")
    print(ranking.to_string(index=False))
    print(f"\nScreen exportado en: {output_root}")


if __name__ == "__main__":
    main()
