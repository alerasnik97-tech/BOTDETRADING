"""
Laboratorio ligero de filtros de contexto sobre una configuracion ya elegida.

Objetivo:
- no reoptimizar
- probar pocas combinaciones simples de weekday/time/direction
- comparar design vs OOS rapido
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import fx_multi_timeframe_backtester as core
import simple_fx_bot as simple


DEFAULT_CONTEXT_VARIANTS: tuple[dict[str, str], ...] = (
    {"label": "base", "weekdays": "", "times": "", "directions": ""},
    {"label": "mon_tue", "weekdays": "mon_tue", "times": "", "directions": ""},
    {"label": "mon_tue_fri", "weekdays": "mon_tue,fri", "times": "", "directions": ""},
    {"label": "mon_tue_early", "weekdays": "mon_tue", "times": "early", "directions": ""},
    {"label": "mon_tue_fri_early", "weekdays": "mon_tue,fri", "times": "early", "directions": ""},
    {"label": "mon_tue_short", "weekdays": "mon_tue", "times": "", "directions": "short"},
    {"label": "mon_tue_fri_short", "weekdays": "mon_tue,fri", "times": "", "directions": "short"},
    {"label": "mon_tue_early_short", "weekdays": "mon_tue", "times": "early", "directions": "short"},
)


def load_base_parameters(summary_path: Path) -> core.StrategyParameters:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return core.StrategyParameters(**payload["parameters"])


def make_run_config(
    *,
    pair: str,
    start: str,
    end: str,
    data_dir: Path,
    report_dir: Path,
    news_file: Path | None,
    strict_data_quality: bool,
) -> core.RunConfig:
    return core.RunConfig(
        start=start,
        end=end,
        pairs=(pair,),
        data_dir=data_dir,
        report_dir=report_dir,
        correlation_groups=core.DEFAULT_CORRELATION_GROUPS,
        source="local",
        download_missing=False,
        force_download=False,
        strict_data_quality=strict_data_quality,
        news_source="csv" if news_file else "none",
        news_file=news_file,
        news_timezone="UTC",
        news_min_importance=3,
        hard_news_veto_enabled=True,
        shock_guard_enabled=True,
        news_no_entry_pre_minutes=45,
        news_no_entry_post_minutes=30,
        news_flatten_minutes_before=15,
        news_hard_no_entry_pre_minutes=90,
        news_hard_no_entry_post_minutes=60,
        news_hard_flatten_minutes_before=30,
        shock_no_entry_atr_multiple=2.5,
        shock_flatten_atr_multiple=3.0,
        shock_cooldown_minutes=30,
    )


def run_stage(
    run_config: core.RunConfig,
    params: core.StrategyParameters,
    broker: core.BrokerConfig,
    *,
    raw_bundle: dict[str, pd.DataFrame],
    news_events: pd.DataFrame,
) -> core.BacktestResult:
    return core.run_backtest(
        run_config,
        params,
        broker,
        raw_bundle=raw_bundle,
        news_events=news_events,
    )


def summarize(stage: str, label: str, result: core.BacktestResult) -> dict[str, Any]:
    summary = result.portfolio_summary
    return {
        "variant": label,
        "stage": stage,
        "total_return_pct": float(summary["total_return_pct"]),
        "max_drawdown_pct": float(summary["max_drawdown_pct"]),
        "profit_factor": float(summary["profit_factor"]),
        "total_trades": int(summary["total_trades"]),
        "trades_per_month": float(summary["trades_per_month"]),
        "win_rate_pct": float(summary["win_rate_pct"]),
        "score": simple.score_result(result),
    }


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    design = summary[summary["stage"] == "design"].add_prefix("design_").rename(columns={"design_variant": "variant"})
    oos = summary[summary["stage"] == "oos"].add_prefix("oos_").rename(columns={"oos_variant": "variant"})
    ranking = design.merge(oos, on="variant", how="inner")
    ranking["accepted"] = (
        (ranking["design_total_return_pct"] >= 0.0)
        & (ranking["design_profit_factor"] >= 1.0)
        & (ranking["oos_total_return_pct"] >= 0.0)
        & (ranking["oos_profit_factor"] >= 1.0)
        & (ranking["oos_trades_per_month"] >= 0.5)
    )
    ranking["lab_score"] = (
        ranking["accepted"].astype(int) * 150.0
        + ranking["oos_score"] * 0.7
        + ranking["design_score"] * 0.3
        + ranking["oos_trades_per_month"] * 10.0
        - ranking["oos_max_drawdown_pct"] * 4.0
    )
    return ranking.sort_values(
        by=["lab_score", "oos_total_return_pct", "oos_profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_decision(ranking: pd.DataFrame) -> dict[str, Any]:
    accepted = ranking[ranking["accepted"]].copy()
    if accepted.empty:
        top = ranking.iloc[0].to_dict() if not ranking.empty else {}
        return {
            "decision": "no_survivor",
            "recommended_variant": top.get("variant"),
        }
    best = accepted.iloc[0]
    return {
        "decision": "survivor_found",
        "recommended_variant": best["variant"],
        "survivors": accepted["variant"].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-lab rapido de filtros de contexto.")
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--design-start", default="2020-01-01")
    parser.add_argument("--design-end", default="2021-12-31")
    parser.add_argument("--design-data-dir", default="data_free_2020/prepared")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--oos-data-dir", default="data_candidates_2022_2025/prepared")
    parser.add_argument("--news-file", default="data/forex_factory_cache.csv")
    parser.add_argument("--report-dir", default="reports_simple_context_lab")
    parser.add_argument("--strict-data-quality", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = args.pair.upper().strip()
    base_summary = Path(args.base_summary)
    news_file = Path(args.news_file) if args.news_file else None
    output_root = core.build_report_dir(Path(args.report_dir), "simple_context_lab")

    design_config = make_run_config(
        pair=pair,
        start=args.design_start,
        end=args.design_end,
        data_dir=Path(args.design_data_dir),
        report_dir=output_root / "design",
        news_file=news_file,
        strict_data_quality=args.strict_data_quality,
    )
    oos_config = make_run_config(
        pair=pair,
        start=args.oos_start,
        end=args.oos_end,
        data_dir=Path(args.oos_data_dir),
        report_dir=output_root / "oos",
        news_file=news_file,
        strict_data_quality=args.strict_data_quality,
    )

    broker = core.BrokerConfig(
        initial_capital=100_000.0,
        risk_fraction=0.01,
        risk_budget_mode="initial_capital",
        commission_rate=0.00002,
        slippage_pips=0.2,
        use_spread_model=True,
        max_leverage=20.0,
        lot_step=1000,
        session_start="11:00",
        session_end="18:45",
    )
    base_params = load_base_parameters(base_summary)
    design_bundle = core.load_raw_bundle(design_config)
    design_news = core.load_news_events(design_config)
    oos_bundle = core.load_raw_bundle(oos_config)
    oos_news = core.load_news_events(oos_config)

    rows: list[dict[str, Any]] = []

    for variant in DEFAULT_CONTEXT_VARIANTS:
        label = variant["label"]
        params = replace(
            base_params,
            context_whitelist_weekdays=variant["weekdays"],
            context_whitelist_times=variant["times"],
            context_whitelist_directions=variant["directions"],
        )
        print(f"\n[simple-context-lab] Variante={label} filtros={variant}")
        design_result = run_stage(design_config, params, broker, raw_bundle=design_bundle, news_events=design_news)
        oos_result = run_stage(oos_config, params, broker, raw_bundle=oos_bundle, news_events=oos_news)

        variant_dir = output_root / label
        simple.export_simple_result(design_result, variant_dir / "design")
        simple.export_simple_result(oos_result, variant_dir / "oos")
        rows.append(summarize("design", label, design_result) | variant)
        rows.append(summarize("oos", label, oos_result) | variant)

    summary = pd.DataFrame(rows)
    ranking = build_ranking(summary)
    decision = build_decision(ranking)

    summary.to_csv(output_root / "context_lab_summary.csv", index=False)
    ranking.to_csv(output_root / "context_lab_ranking.csv", index=False)
    (output_root / "context_lab_decision.json").write_text(
        json.dumps(core.sanitize_for_json(decision), indent=2),
        encoding="utf-8",
    )

    print("\\n=== SIMPLE CONTEXT LAB SUMMARY ===")
    print(summary.to_string(index=False))
    print("\\n=== SIMPLE CONTEXT LAB RANKING ===")
    print(ranking.to_string(index=False))
    print("\\n=== SIMPLE CONTEXT LAB DECISION ===")
    print(json.dumps(core.sanitize_for_json(decision), indent=2))
    print(f"\\nReportes exportados en: {output_root}")


if __name__ == "__main__":
    main()
