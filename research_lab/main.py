from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab.config import (
    ALT_WFA_IS_MONTHS,
    ALT_WFA_OOS_MONTHS,
    DEFAULT_DATA_DIRS,
    DEFAULT_MAX_EVALS_PER_STRATEGY,
    DEFAULT_NEWS_V2_UTC_FILE,
    DEFAULT_PAIR,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_COST_PROFILE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEED,
    DEFAULT_WFA_IS_MONTHS,
    DEFAULT_WFA_OOS_MONTHS,
    EngineConfig,
    INITIAL_CAPITAL,
    NewsConfig,
    STRATEGY_NAMES,
    SessionConfig,
    SESSION_VARIANTS,
    SUPPORTED_COST_PROFILES,
    SUPPORTED_EXECUTION_MODES,
    SUPPORTED_INTRABAR_POLICIES,
    resolved_cost_profile,
    resolved_intrabar_policy,
    with_execution_mode,
)
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, load_news_events, news_result_payload
from research_lab.plotting import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_heatmap,
    plot_overlay_drawdown,
    plot_overlay_equity,
    plot_overlay_yearly,
    plot_yearly_pnl,
)
from research_lab.report import export_root_tables, export_strategy_bundle, summarize_result, sync_visible_chatgpt
from research_lab.scorer import compute_final_score, score_is_summary
from research_lab.strategies import STRATEGY_REGISTRY
from research_lab.validation import parameter_combinations, run_default_and_alt_wfa, WFAResult
from research_lab.rejection_protocol import evaluate_is_rejection, evaluate_oos_rejection, HARD_REJECT, SOFT_REJECT, PASS_MINIMUM, STRONG_CANDIDATE
def build_output_root(results_dir: Path, label: str) -> Path:
    timestamp = pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_{label}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def schedule_from_params(params: dict[str, Any]) -> dict[str, str]:
    session_name = params.get("session_name")
    if session_name in SESSION_VARIANTS:
        entry_start, entry_end = SESSION_VARIANTS[session_name]
    else:
        session = SessionConfig()
        entry_start, entry_end = session.entry_start, session.entry_end
    return {"entry_start": entry_start, "entry_end": entry_end, "force_close": SessionConfig().force_close}


def costs_payload(engine_config: EngineConfig) -> dict[str, Any]:
    return {
        "assumed_spread_pips": engine_config.assumed_spread_pips if engine_config.assumed_spread_pips is not None else engine_config.max_spread_pips,
        "max_allowed_spread_pips": engine_config.max_spread_pips,
        "slippage_pips": engine_config.slippage_pips,
        "commission_per_lot_roundturn_usd": engine_config.commission_per_lot_roundturn_usd,
        "risk_pct": engine_config.risk_pct,
        "initial_capital": INITIAL_CAPITAL,
        "price_source": engine_config.price_source,
        "opening_session_end": engine_config.opening_session_end,
        "late_session_start": engine_config.late_session_start,
        "spread_opening_multiplier": engine_config.spread_opening_multiplier,
        "high_vol_range_atr": engine_config.high_vol_range_atr,
        "spread_high_vol_multiplier": engine_config.spread_high_vol_multiplier,
        "spread_late_session_multiplier": engine_config.spread_late_session_multiplier,
        "slippage_opening_multiplier": engine_config.slippage_opening_multiplier,
        "slippage_high_vol_multiplier": engine_config.slippage_high_vol_multiplier,
        "slippage_stop_multiplier": engine_config.slippage_stop_multiplier,
        "slippage_target_multiplier": engine_config.slippage_target_multiplier,
        "slippage_late_session_multiplier": engine_config.slippage_late_session_multiplier,
        "slippage_forced_close_multiplier": engine_config.slippage_forced_close_multiplier,
        "slippage_final_close_multiplier": engine_config.slippage_final_close_multiplier,
        "stress_spread_multiplier": engine_config.stress_spread_multiplier,
        "stress_slippage_multiplier": engine_config.stress_slippage_multiplier,
        "ambiguity_slippage_multiplier": engine_config.ambiguity_slippage_multiplier,
        "intrabar_exit_priority": engine_config.intrabar_exit_priority,
        "execution_mode": engine_config.execution_mode,
        "cost_profile_used": resolved_cost_profile(engine_config),
        "intrabar_policy_used": resolved_intrabar_policy(engine_config),
    }


def yearly_positive_count(yearly_stats: pd.DataFrame) -> int:
    if yearly_stats.empty:
        return 0
    return int((yearly_stats.groupby("year")["total_pnl_r"].sum() > 0).sum())


def share_best_year(yearly_stats: pd.DataFrame) -> float:
    if yearly_stats.empty:
        return 0.0
    yearly = yearly_stats.groupby("year")["total_pnl_r"].sum()
    positive_total = float(yearly[yearly > 0].sum())
    if positive_total <= 0:
        return 0.0
    return float(yearly.max() / positive_total)


def plateau_metrics(optimization_df: pd.DataFrame) -> tuple[float, float]:
    if optimization_df.empty or "support_score" not in optimization_df.columns:
        return 0.0, 1.0
    count = max(1, int(np.ceil(len(optimization_df) * 0.10)))
    top = optimization_df.nlargest(count, "support_score")
    best = float(top["support_score"].max())
    median = float(top["support_score"].median())
    plateau_index = len(top) / max(len(optimization_df), 1)
    gap = best - median
    return plateau_index, gap


def strategy_row(
    strategy_name: str,
    summary: dict[str, Any],
    oos_summary: dict[str, Any],
    alt_oos_summary: dict[str, Any],
    selected_score: float,
) -> dict[str, Any]:
    return {
        "strategy_name": strategy_name,
        "total_trades": summary["total_trades"],
        "avg_trades_per_month": summary["avg_trades_per_month"],
        "win_rate": summary["win_rate"],
        "breakeven_rate": summary["breakeven_rate"],
        "profit_factor": summary["profit_factor"],
        "expectancy_r": summary["expectancy_r"],
        "total_return_pct": summary["total_return_pct"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "negative_months": summary["negative_months"],
        "negative_years": summary["negative_years"],
        "insufficient_sample": summary["insufficient_sample"],
        "sample_penalty_applied": summary["sample_penalty_applied"],
        "pf_oos": oos_summary["profit_factor"],
        "expectancy_oos": oos_summary["expectancy_r"],
        "dd_oos": oos_summary["max_drawdown_pct"],
        "return_oos": oos_summary["total_return_pct"],
        "years_positive_oos": 4 - int(oos_summary["negative_years"]),
        "trades_mes_oos": oos_summary["avg_trades_per_month"],
        "pf_oos_alt": alt_oos_summary["profit_factor"],
        "expectancy_oos_alt": alt_oos_summary["expectancy_r"],
        "dd_oos_alt": alt_oos_summary["max_drawdown_pct"],
        "selected_score": selected_score,
        "parameter_set_used": json.dumps(summary["parameter_set_used"], ensure_ascii=False),
    }


def evaluate_strategy(
    strategy_name: str,
    frame: pd.DataFrame,
    engine_config: EngineConfig,
    news_config: NewsConfig,
    output_root: Path,
    max_evals: int,
    seed: int,
    precision_package: dict[str, pd.DataFrame] | None = None,
    data_source_used: str | None = None,
    timeframe: str = "M15",
) -> dict[str, Any]:
    strategy_module = STRATEGY_REGISTRY[strategy_name]
    combos = parameter_combinations(strategy_module, max_evals=max_evals, seed=seed)
    news_result = load_news_events(engine_config.pair, news_config)
    news_filter_used = news_result.enabled
    news_block = build_entry_block(entry_open_index(frame.index), news_result.events, news_config)
    strategy_dir = output_root / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------
    # METADATA & LINAJE CANONICO OBLIGATORIO
    # ---------------------------------------------
    import datetime
    lineage_metadata = {
        "strategy_name": strategy_name,
        "runner_used": "main.py (Research Lab F1)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine_config": {
            "pair": engine_config.pair,
            "execution_mode": engine_config.execution_mode,
            "cost_profile": engine_config.cost_profile,
            "intrabar_policy": engine_config.intrabar_policy,
            "risk_pct": engine_config.risk_pct
        },
        "dataset_used": data_source_used,
        "is_months": combos[0].get("wfa_is_months", 24) if combos else 24,
        "oos_months": combos[0].get("wfa_oos_months", 6) if combos else 6,
        "news_filter_enabled": news_config.enabled,
        "versions": {}
    }
    
    try:
        from research_lab.version import LAB_VERSION, CANONICAL_CONTRACT_VERSION, REJECTION_PROTOCOL_VERSION, COST_MODEL_VERSION, STRATEGY_PROMOTION_POLICY_VERSION
        lineage_metadata["versions"] = {
            "lab": LAB_VERSION,
            "contract": CANONICAL_CONTRACT_VERSION,
            "rejection_protocol": REJECTION_PROTOCOL_VERSION,
            "cost_model": COST_MODEL_VERSION,
            "promotion_policy": STRATEGY_PROMOTION_POLICY_VERSION
        }
    except ImportError:
        pass
        
    # GUARDRAIL EXPLÍCITO DE MODO DE EJECUCION
    if engine_config.execution_mode == "normal":
        lineage_metadata["guardrail_warning"] = "NORMAL_MODE_ACTIVE: Los fills asumen el mejor de los casos (cruzando High/Low de M5). Vulnerable a espejismos intradiarios. NO USAR para Capital Real sin validar con 'stress' o intrabar_policy='worst_case'."
    elif engine_config.execution_mode == "stress":
        lineage_metadata["guardrail_warning"] = "STRESS_MODE_ACTIVE: Fills pesimistas activados. Resultados pueden ser confiables si el edge sobrevive, pero el modo 'precision' es preferible si la estrategia es de scalp (RR < 1.0)."
    elif engine_config.execution_mode == "precision":
        lineage_metadata["guardrail_warning"] = "PRECISION_MODE_ACTIVE: Tick-level simulation on Bid/Ask spreads. Highest fidelity execution. Resultados tomados como SERIOS."
        
    with open(strategy_dir / "lineage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lineage_metadata, f, indent=4)
        
    optimization_rows: list[dict[str, Any]] = []
    best_support_score = -float("inf")
    best_payload: dict[str, Any] | None = None

    for params in combos:
        result = run_backtest(
            strategy_module,
            frame,
            params,
            engine_config,
            news_block,
            news_filter_used,
            precision_package=precision_package,
            data_source_used=data_source_used,
        )
        summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
            strategy_name,
            result.trades,
            result.equity_curve,
            params,
            news_filter_used,
            INITIAL_CAPITAL,
            None,
            costs_payload(engine_config),
            timeframe,
            schedule_from_params(params),
            params.get("break_even_at_r"),
        )
        support_score = score_is_summary(summary)
        row = {
            **summary,
            "support_score": support_score,
            "parameter_set_used": json.dumps(params, ensure_ascii=False),
        }
        for key, value in params.items():
            row[key] = value
        optimization_rows.append(row)
        if support_score > best_support_score:
            best_support_score = support_score
            best_payload = {
                "params": params,
                "result": result,
                "summary": summary,
                "trades_export": trades_export,
                "monthly_stats": monthly_stats,
                "yearly_stats": yearly_stats,
                "equity_export": equity_export,
            }

    if best_payload is None:
        raise RuntimeError(f"No pude evaluar {strategy_name}")

    optimization_df = pd.DataFrame(optimization_rows).sort_values("support_score", ascending=False).reset_index(drop=True)
    
    # NEW HARNESS: REJECTION PROTOCOL (IN-SAMPLE SHORT-CIRCUIT)
    best_is_summary = optimization_df.iloc[0].to_dict() if not optimization_df.empty else {}
    is_rejected, is_level, is_reason = evaluate_is_rejection(best_is_summary)
    
    # Update lineage with IS decision
    lineage_metadata["is_rejection_level"] = is_level
    lineage_metadata["is_rejection_reason"] = is_reason
    lineage_metadata["final_promotion_status"] = is_level if is_rejected else "PENDING_OOS"
    
    with open(strategy_dir / "lineage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lineage_metadata, f, indent=4)
        
    if is_rejected:
        print(f"[{strategy_name}] ABORTADO IN-SAMPLE: {is_reason} ({is_level})")
        with open(strategy_dir / "REJECTION_REPORT.md", "w", encoding="utf-8") as f:
            f.write(f"# ESTRATEGIA RECHAZADA TEMPRANAMENTE\nNivel: {is_level}\nFase: IN-SAMPLE\nMotivo: {is_reason}\n\nNo se ejecutó WFA para ahorrar CPU. El mejor IS Profit Factor fue {best_is_summary.get('profit_factor', 0)} y Expectancy_R {best_is_summary.get('expectancy_r', 0)}.")
        return {
            "strategy_name": strategy_name,
            "status": is_level,
            "rejection_reason": is_reason,
            "row": {"strategy_name": strategy_name, "insufficient_sample": True, "selected_score": -9999, "pf_oos": 0, "expectancy_oos": 0, "dd_oos": 0, "trades_mes_oos": 0, "return_oos": 0, "years_positive_oos": 0, "parameter_set_used": "{}"}
        }

    default_wfa, alt_wfa = run_default_and_alt_wfa(
        strategy_name=strategy_name,
        strategy_module=strategy_module,
        frame=frame,
        combos=combos,
        engine_config=engine_config,
        news_config=news_config,
        precision_package=precision_package,
        data_source_used=data_source_used,
    )

    plateau_index, top10_gap = plateau_metrics(optimization_df)
    positive_years_full = yearly_positive_count(best_payload["yearly_stats"])
    share_year = share_best_year(best_payload["yearly_stats"])
    selected_score = compute_final_score(
        full_summary=best_payload["summary"],
        oos_summary=default_wfa.oos_summary,
        plateau_index=plateau_index,
        top10_median_gap=top10_gap,
        positive_years_full=positive_years_full,
        share_best_year=share_year,
    )

    summary = dict(best_payload["summary"])
    summary["selected_score"] = selected_score
    summary["wfa_default"] = {
        "is_months": DEFAULT_WFA_IS_MONTHS,
        "oos_months": DEFAULT_WFA_OOS_MONTHS,
        "profit_factor": default_wfa.oos_summary["profit_factor"],
        "expectancy_r": default_wfa.oos_summary["expectancy_r"],
        "max_drawdown_pct": default_wfa.oos_summary["max_drawdown_pct"],
        "total_return_pct": default_wfa.oos_summary["total_return_pct"],
        "avg_trades_per_month": default_wfa.oos_summary["avg_trades_per_month"],
    }
    summary["wfa_alt"] = {
        "is_months": ALT_WFA_IS_MONTHS,
        "oos_months": ALT_WFA_OOS_MONTHS,
        "profit_factor": alt_wfa.oos_summary["profit_factor"],
        "expectancy_r": alt_wfa.oos_summary["expectancy_r"],
        "max_drawdown_pct": alt_wfa.oos_summary["max_drawdown_pct"],
        "total_return_pct": alt_wfa.oos_summary["total_return_pct"],
        "avg_trades_per_month": alt_wfa.oos_summary["avg_trades_per_month"],
    }
    summary["plateau_index"] = plateau_index
    summary["top10_median_gap"] = top10_gap
    summary["share_best_year"] = share_year
    summary["news_module"] = news_result_payload(news_result)

    strategy_dir = output_root / strategy_name
    export_strategy_bundle(
        strategy_dir / "PARA CHATGPT",
        summary=summary,
        trades_export=best_payload["trades_export"],
        monthly_stats=best_payload["monthly_stats"],
        yearly_stats=best_payload["yearly_stats"],
        equity_export=best_payload["equity_export"],
        optimization_results=optimization_df,
        extra_frames={
            "walkforward_default.csv": default_wfa.fold_rows,
            "walkforward_alt.csv": alt_wfa.fold_rows,
            "wfa_default_equity_curve.csv": summarize_result(
                strategy_name,
                default_wfa.oos_trades,
                default_wfa.oos_equity_curve,
                {"wfa": "default"},
                news_filter_used,
                INITIAL_CAPITAL,
                None,
            )[4],
        },
    )

    plot_equity_curve(best_payload["equity_export"], strategy_dir / "PARA CHATGPT" / "equity_curve.png", f"{strategy_name} equity")
    plot_drawdown_curve(best_payload["equity_export"], strategy_dir / "PARA CHATGPT" / "drawdown_curve.png", f"{strategy_name} drawdown")
    plot_yearly_pnl(best_payload["yearly_stats"], strategy_dir / "PARA CHATGPT" / "yearly_pnl_r_bar.png", f"{strategy_name} yearly pnl_r")
    plot_heatmap(optimization_df, strategy_dir / "PARA CHATGPT" / "parameter_sensitivity_heatmaps.png", f"{strategy_name} sensitivity")

    # NEW HARNESS: OOS REJECTION
    oos_rejected, oos_level, oos_reason = evaluate_oos_rejection(default_wfa.oos_summary, bool(summary.get("insufficient_sample", False)))
    if oos_rejected:
        print(f"[{strategy_name}] DESCARTADA POST-WFA OOS: {oos_reason} ({oos_level})")
        with open(strategy_dir / "REJECTION_REPORT.md", "w", encoding="utf-8") as f:
            f.write(f"# ESTRATEGIA RECHAZADA OOS\nNivel: {oos_level}\nFase: OUT-OF-SAMPLE WFA\nMotivo: {oos_reason}\n\nEsta estrategia completó WFA pero falló los umbrales mínimos de robustez.")
    # Update lineage with OOS decision & final canonical promotion mapping
    lineage_metadata["oos_rejection_level"] = oos_level
    lineage_metadata["oos_rejection_reason"] = oos_reason
    lineage_metadata["final_promotion_status"] = oos_level
    
    with open(strategy_dir / "lineage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lineage_metadata, f, indent=4)
        
    return {
        "strategy_name": strategy_name,
        "status": oos_level,
        "rejection_reason": oos_reason,
        "summary": summary,
        "optimization_df": optimization_df,
        "default_wfa": default_wfa,
        "alt_wfa": alt_wfa,
        "yearly_stats": best_payload["yearly_stats"],
        "equity_export": best_payload["equity_export"],
        "row": strategy_row(strategy_name, summary, default_wfa.oos_summary, alt_wfa.oos_summary, selected_score),
    }


def build_top3_markdown(ranking: pd.DataFrame) -> str:
    lines = ["# Top 3 finalistas", ""]
    for _, row in ranking.head(3).iterrows():
        lines.append(f"## {row['strategy_name']}")
        lines.append(f"- score: {row['selected_score']:.2f}")
        lines.append(f"- PF OOS: {row['pf_oos']:.4f}")
        lines.append(f"- expectancy OOS: {row['expectancy_oos']:.4f}R")
        lines.append(f"- DD OOS: {row['dd_oos']:.2f}%")
        lines.append(f"- trades/mes OOS: {row['trades_mes_oos']:.2f}")
        lines.append(f"- insufficient_sample: {row['insufficient_sample']}")
        lines.append("")
    return "\n".join(lines)


def build_losers_markdown(ranking: pd.DataFrame) -> str:
    lines = ["# Autopsia perdedores", ""]
    for _, row in ranking.iloc[3:].iterrows():
        lines.append(f"## {row['strategy_name']}")
        reasons: list[str] = []
        if row["insufficient_sample"]:
            reasons.append("muestra insuficiente")
        if row["pf_oos"] < 1.0:
            reasons.append("PF OOS < 1")
        if row["expectancy_oos"] <= 0:
            reasons.append("expectancy OOS <= 0")
        if row["dd_oos"] > 10:
            reasons.append("drawdown OOS alto")
        lines.append(f"- causas: {', '.join(reasons) if reasons else 'sin edge defendible'}")
        lines.append("")
    return "\n".join(lines)


def build_recommendation_markdown(ranking: pd.DataFrame) -> str:
    lines = ["# Recomendacion final", ""]
    if ranking.empty:
        lines.append("No hay resultados.")
        return "\n".join(lines)
    winner = ranking.iloc[0]
    candidate = (
        not bool(winner["insufficient_sample"])
        and float(winner["pf_oos"]) > 1.15
        and float(winner["expectancy_oos"]) > 0.05
        and float(winner["dd_oos"]) < 10.0
        and int(winner["years_positive_oos"]) >= 4
    )
    if candidate:
        lines.append(f"Estrategia final: **{winner['strategy_name']}**")
        lines.append("")
        lines.append("- Cumple los umbrales mínimos OOS.")
    else:
        lines.append("Ninguna estrategia cumple los umbrales finales.")
        lines.append("")
        lines.append(f"Mejor compromiso actual: **{winner['strategy_name']}**")
        lines.append("- Sirve como baseline de la próxima iteración, no como sistema final.")
    lines.append("")
    lines.append("Siguiente iteración sugerida:")
    lines.append(f"- trabajar solo sobre {winner['strategy_name']} con ajustes pequeños y medibles")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laboratorio robusto M15 para EURUSD.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--pair", default=DEFAULT_PAIR)
        target.add_argument("--start", default="2020-01-01")
        target.add_argument("--end", default="2025-12-31")
        target.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
        target.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
        target.add_argument("--news-file", default=str(DEFAULT_NEWS_V2_UTC_FILE))
        target.add_argument("--disable-news", action="store_true")
        target.add_argument("--news-pre-minutes", type=int, default=15)
        target.add_argument("--news-post-minutes", type=int, default=15)
        target.add_argument("--risk-pct", type=float, default=0.5)
        target.add_argument("--shock-candle-atr-max", type=float, default=2.2)
        target.add_argument("--assumed-spread-pips", type=float, default=1.2)
        target.add_argument("--max-spread-pips", type=float, default=1.2)
        target.add_argument("--slippage-pips", type=float, default=0.2)
        target.add_argument("--commission-per-lot-roundturn-usd", type=float, default=7.0)
        target.add_argument("--execution-mode", choices=list(SUPPORTED_EXECUTION_MODES), default=DEFAULT_EXECUTION_MODE)
        target.add_argument("--cost-profile", choices=list(SUPPORTED_COST_PROFILES), default="auto")
        target.add_argument("--intrabar-policy", choices=list(SUPPORTED_INTRABAR_POLICIES), default="auto")
        target.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS_PER_STRATEGY)
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--max-trades-per-day", type=int, default=2)

    run_parser = subparsers.add_parser("run", help="Corre una estrategia con parámetros por defecto.")
    add_common(run_parser)
    run_parser.add_argument("--strategy", choices=STRATEGY_NAMES, required=True)

    optimize_parser = subparsers.add_parser("optimize", help="Optimiza una estrategia con WFA.")
    add_common(optimize_parser)
    optimize_parser.add_argument("--strategy", choices=STRATEGY_NAMES, required=True)

    all_parser = subparsers.add_parser("run-all", help="Corre todas las estrategias y genera ranking.")
    add_common(all_parser)
    parser.add_argument("--timeframe", type=str, choices=["M5", "M15"], default="M15", help="Timeframe de resolución para el laboratorio.")
    return parser


def build_configs(args: argparse.Namespace) -> tuple[EngineConfig, NewsConfig]:
    engine_config = with_execution_mode(
        EngineConfig(
        pair=args.pair.upper().strip(),
        risk_pct=args.risk_pct,
        shock_candle_atr_max=args.shock_candle_atr_max,
        assumed_spread_pips=args.assumed_spread_pips,
        max_spread_pips=args.max_spread_pips,
        commission_per_lot_roundturn_usd=args.commission_per_lot_roundturn_usd,
        slippage_pips=args.slippage_pips,
        execution_mode=args.execution_mode,
        cost_profile=args.cost_profile,
        intrabar_policy=args.intrabar_policy,
        max_trades_per_day=args.max_trades_per_day,
        ),
        args.execution_mode,
    )
    news_config = NewsConfig(
        enabled=not args.disable_news,
        file_path=Path(args.news_file),
        source_approved=True,
        pre_minutes=args.news_pre_minutes,
        post_minutes=args.news_post_minutes,
    )
    return engine_config, news_config


def print_ranking_console_summary(ranking: pd.DataFrame, runtime_seconds: float) -> None:
    print("\n=== STRATEGY RANKING ===")
    if ranking.empty:
        print("Sin resultados.")
    else:
        print(ranking.to_string(index=False))
    print(f"\nruntime_seconds: {runtime_seconds:.2f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    engine_config, news_config = build_configs(args)
    data_bundle = load_backtest_data_bundle(
        engine_config.pair,
        [Path(item) for item in args.data_dirs],
        args.start,
        args.end,
        engine_config.execution_mode,
        target_timeframe=args.timeframe,
    )
    frame = data_bundle.frame
    output_root = build_output_root(Path(args.results_dir), "robust_lab")
    start_time = time.time()

    if args.command in {"run", "optimize"}:
        strategy_names = [args.strategy]
    else:
        strategy_names = list(STRATEGY_NAMES)

    evaluations = [
        evaluate_strategy(
            strategy_name,
            frame,
            engine_config,
            news_config,
            output_root,
            args.max_evals,
            args.seed,
            precision_package=data_bundle.precision_package,
            data_source_used=data_bundle.data_source_used,
            timeframe=args.timeframe,
        )
        for strategy_name in strategy_names
    ]

    ranking = pd.DataFrame([item["row"] for item in evaluations]).sort_values(
        ["insufficient_sample", "selected_score"], ascending=[True, False]
    ).reset_index(drop=True)

    comparative_table = ranking[
        [
            "strategy_name",
            "pf_oos",
            "expectancy_oos",
            "dd_oos",
            "return_oos",
            "years_positive_oos",
            "trades_mes_oos",
            "selected_score",
            "parameter_set_used",
        ]
    ].copy()

    top3_curves = {}
    top3_yearly = {}
    for _, row in ranking.head(3).iterrows():
        item = next(entry for entry in evaluations if entry["strategy_name"] == row["strategy_name"])
        if "default_wfa" not in item:
            continue
        _, _, _, yearly_stats_oos, equity_export_oos = summarize_result(
            item["strategy_name"],
            item["default_wfa"].oos_trades,
            item["default_wfa"].oos_equity_curve,
            {"wfa": "default"},
            True,
            INITIAL_CAPITAL,
            None,
        )
        top3_curves[item["strategy_name"]] = equity_export_oos
        top3_yearly[item["strategy_name"]] = yearly_stats_oos

    plot_overlay_equity(top3_curves, output_root / "equity_curves_overlay_top3.png", "Top3 OOS equity")
    plot_overlay_drawdown(top3_curves, output_root / "drawdown_overlay_top3.png", "Top3 OOS drawdown")
    plot_overlay_yearly(top3_yearly, output_root / "yearly_pnl_overlay_top3.png", "Top3 OOS yearly pnl_r")

    # Collect rejection stats for main report
    rejection_log = []
    for item in evaluations:
        status = item.get("status", "N/A")
        reason = item.get("rejection_reason", "N/A")
        rejection_log.append({ "strategy_name": item["strategy_name"], "status": status, "reason": reason })
    pd.DataFrame(rejection_log).to_csv(output_root / "rejection_summary_log.csv", index=False)

    export_root_tables(
        output_root,
        ranking,
        comparative_table,
        build_top3_markdown(ranking),
        build_losers_markdown(ranking),
        build_recommendation_markdown(ranking),
    )
    sync_visible_chatgpt(output_root)
    print_ranking_console_summary(ranking, time.time() - start_time)


if __name__ == "__main__":
    main()
