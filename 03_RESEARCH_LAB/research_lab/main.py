from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import (
    DEFAULT_DATA_DIRS,
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    DEFAULT_NEWS_V2_UTC_FILE,
    DEFAULT_PAIR,
    DEFAULT_RESULTS_DIR,
    INITIAL_CAPITAL,
    STRATEGY_NAMES,
    SUPPORTED_COST_PROFILES,
    SUPPORTED_EXECUTION_MODES,
    SUPPORTED_INTRABAR_POLICIES,
    DEFAULT_MAX_EVALS_PER_STRATEGY,
    DEFAULT_SEED,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_RISK_PCT,
    DEFAULT_SPREAD_PIPS,
    DEFAULT_SLIPPAGE_PIPS,
    DEFAULT_COMMISSION_ROUNDTURN_USD,
    EngineConfig,
    NewsConfig,
    with_execution_mode,
)
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.report import (
    export_root_tables,
    summarize_result,
    sync_visible_chatgpt,
)
from research_lab.rejection_protocol import apply_rejection_logic
from research_lab.strategies import STRATEGY_REGISTRY
from research_lab.wfa import run_wfa_default


def build_output_root(base_dir: Path, label: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = base_dir / f"{timestamp}_{label}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def evaluate_strategy(
    name: str,
    data: Any,
    engine_config: Any,
    news_config: Any,
    output_root: Path,
    max_evals: int,
    seed: int,
    *,
    precision_package: Any = None,
    data_source_used: str | None = None,
    timeframe: str = "M15",
    fixed_params: dict | None = None,
) -> dict[str, Any]:
    strategy_module = STRATEGY_REGISTRY.get(name)
    if not strategy_module:
        raise ValueError(f"Estrategia '{name}' no encontrada en el registro.")

    # 1. Ejecutar WFA (usando parametros fijos si se proveen)
    wfa_res = run_wfa_default(
        strategy_module,
        data.frame,
        engine_config,
        news_config,
        max_evals=max_evals if fixed_params is None else 1,
        seed=seed,
        precision_package=precision_package,
        data_source_used=data_source_used,
        fixed_params=fixed_params,
    )

    # 2. Aplicar Protocolo de Rechazo (In-Sample + Out-of-Sample)
    status, rejection_reason, score = apply_rejection_logic(wfa_res)

    # 3. Preparar fila del ranking
    ranking_row = {
        "strategy_name": name,
        "insufficient_sample": wfa_res.insufficient_sample,
        "selected_score": score,
        "pf_oos": wfa_res.oos_stats.get("profit_factor", 0.0),
        "expectancy_oos": wfa_res.oos_stats.get("expectancy_r", 0.0),
        "dd_oos": wfa_res.oos_stats.get("max_drawdown_pct", 0.0),
        "trades_mes_oos": wfa_res.oos_stats.get("avg_trades_per_month", 0.0),
        "return_oos": wfa_res.oos_stats.get("total_return_pct", 0.0),
        "years_positive_oos": wfa_res.oos_stats.get("years_positive", 0),
        "parameter_set_used": json.dumps(wfa_res.best_params),
    }

    # 4. Exportar bundle de la estrategia
    strategy_dir = output_root / name
    from research_lab.report import export_strategy_bundle
    
    # Generamos los DF de reporte para el mejor set
    summary, trades_exp, monthly, yearly, equity = summarize_result(
        name,
        wfa_res.oos_trades,
        wfa_res.oos_equity_curve,
        wfa_res.best_params,
        True,
        INITIAL_CAPITAL,
        score,
        timeframe=timeframe,
    )

    export_strategy_bundle(
        strategy_dir,
        summary=summary,
        trades_export=trades_exp,
        monthly_stats=monthly,
        yearly_stats=yearly,
        equity_export=equity,
        optimization_results=wfa_res.optimization_results,
        extra_frames={},
        extra_json={"lineage_metadata.json": wfa_res.lineage},
    )

    if status != "pass":
        (strategy_dir / "REJECTION_REPORT.md").write_text(
            f"# ESTRATEGIA RECHAZADA TEMPRANAMENTE\nNivel: {status}\nFase: {'IN-SAMPLE' if 'IS' in rejection_reason else 'OUT-OF-SAMPLE'}\nMotivo: {rejection_reason}\n\n"
            f"No se ejecutó WFA completo para ahorrar CPU si falló el primer IS. El mejor IS Profit Factor fue {wfa_res.best_is_pf:.2f} y Expectancy_R {wfa_res.best_is_expectancy:.2f}."
        )

    return {
        "strategy_name": name,
        "row": ranking_row,
        "status": status,
        "rejection_reason": rejection_reason,
        "default_wfa": wfa_res,
    }


def build_top3_markdown(ranking: pd.DataFrame) -> str:
    lines = ["# Top 3 Finalistas (Consolidado OOS)", ""]
    if ranking.empty:
        lines.append("No hay estrategias que hayan pasado los filtros.")
        return "\n".join(lines)
    for i, (_, row) in enumerate(ranking.head(3).iterrows()):
        lines.append(f"{i+1}. **{row['strategy_name']}**")
        lines.append(f"   - Score: {row['selected_score']:.2f}")
        lines.append(f"   - PF OOS: {row['pf_oos']:.2f} | Expectancy: {row['expectancy_oos']:.3f}")
        lines.append(f"   - Trades/mes: {row['trades_mes_oos']:.1f}")
        lines.append("")
    return "\n".join(lines)


def build_losers_markdown(ranking: pd.DataFrame) -> str:
    lines = ["# Autopsia de Perdedores / Rechazados", ""]
    losers = ranking[ranking["selected_score"] < -500].copy()
    if losers.empty:
        lines.append("Todas las estrategias evaluadas mostraron algun edge.")
        return "\n".join(lines)
    for _, row in losers.iterrows():
        lines.append(f"### {row['strategy_name']}")
        reasons = []
        if bool(row["insufficient_sample"]):
            reasons.append("muestra insuficiente")
        if row["pf_oos"] < 1:
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
        target.add_argument("--news-pre-minutes", type=int, default=30)
        target.add_argument("--news-post-minutes", type=int, default=60)
        target.add_argument("--risk-pct", type=float, default=DEFAULT_RISK_PCT)
        target.add_argument("--shock-candle-atr-max", type=float, default=2.2)
        target.add_argument("--assumed-spread-pips", type=float, default=DEFAULT_SPREAD_PIPS)
        target.add_argument("--max-spread-pips", type=float, default=3.0)
        target.add_argument("--slippage-pips", type=float, default=DEFAULT_SLIPPAGE_PIPS)
        target.add_argument("--commission-per-lot-roundturn-usd", type=float, default=DEFAULT_COMMISSION_ROUNDTURN_USD)
        target.add_argument("--execution-mode", choices=list(SUPPORTED_EXECUTION_MODES), default=DEFAULT_EXECUTION_MODE)
        target.add_argument("--cost-profile", choices=list(SUPPORTED_COST_PROFILES), default="auto")
        target.add_argument("--intrabar-policy", choices=list(SUPPORTED_INTRABAR_POLICIES), default="auto")
        target.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS_PER_STRATEGY)
        target.add_argument("--seed", type=int, default=DEFAULT_SEED)
        target.add_argument("--max-trades-per-day", type=int, default=2)
        target.add_argument("--fomc-only", action="store_true", help="Filtrar solo eventos FOMC/FED.")
        target.add_argument("--session-cutoff", type=str, help="Forzar cierre de sesión a esta hora (HH:MM).")

    run_parser = subparsers.add_parser("run", help="Corre una estrategia con parámetros por defecto.")
    add_common(run_parser)
    run_parser.add_argument("--strategy", choices=STRATEGY_NAMES, required=True)
    run_parser.add_argument("--params", type=str, help="Parametros en formato JSON para bypass de optimización.")

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
        session_cutoff=args.session_cutoff,
        ),
        args.execution_mode,
    )
    news_config = NewsConfig(
        enabled=not args.disable_news,
        file_path=Path(args.news_file),
        source_approved=True,
        pre_minutes=args.news_pre_minutes,
        post_minutes=args.news_post_minutes,
        fomc_only=args.fomc_only,
    )
    return engine_config, news_config


def print_ranking_console_summary(ranking: pd.DataFrame, runtime_seconds: float) -> None:
    print("\n=== STRATEGY RANKING ===")
    if ranking.empty:
        print("Sin resultados.")
    else:
        print(ranking.to_string(index=False))
    print(f"\nruntime_seconds: {runtime_seconds:.2f}")


def plot_overlay_equity(curves: dict, path: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for name, df in curves.items():
        if df.empty: continue
        plt.plot(pd.to_datetime(df["datetime_ny"]), df["equity"], label=name)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_overlay_drawdown(curves: dict, path: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for name, df in curves.items():
        if df.empty: continue
        plt.plot(pd.to_datetime(df["datetime_ny"]), df["drawdown_pct"], label=name)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_overlay_yearly(stats: dict, path: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    for name, df in stats.items():
        if df.empty: continue
        plt.bar(df["year"], df["total_pnl_r"], alpha=0.5, label=name)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()


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
    output_root = build_output_root(Path(args.results_dir), "robust_lab")
    start_time = time.time()

    if args.command in {"run", "optimize"}:
        strategy_names = [args.strategy]
    else:
        strategy_names = list(STRATEGY_NAMES)

    fixed_params = None
    if args.command == "run" and args.params:
        try:
            fixed_params = json.loads(args.params.replace("'", "\""))
        except Exception as e:
            print(f"Error parseando params: {e}")

    evaluations = []
    for i, strategy_name in enumerate(strategy_names):
        print(f"--- Evaluando {strategy_name} ({i+1}/{len(strategy_names)}) ---")
        try:
            res = evaluate_strategy(
                strategy_name,
                data_bundle,
                engine_config,
                news_config,
                output_root,
                args.max_evals,
                args.seed,
                precision_package=data_bundle.precision_package,
                data_source_used=data_bundle.data_source_used,
                timeframe=args.timeframe,
                fixed_params=fixed_params,
            )
            evaluations.append(res)
        except Exception as e:
            print(f"Error en {strategy_name}: {e}")

    if not evaluations:
        print("\n[WARNING] No se completó ninguna evaluación con éxito. Verifique los errores arriba.")
        return

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
        item = next((entry for entry in evaluations if entry["strategy_name"] == row["strategy_name"]), None)
        if not item or "default_wfa" not in item:
            continue
        from research_lab.report import summarize_result
        _, _, _, yearly_stats_oos, equity_export_oos = summarize_result(
            item["strategy_name"],
            item["default_wfa"].oos_trades,
            item["default_wfa"].oos_equity_curve,
            item["default_wfa"].best_params,
            True,
            INITIAL_CAPITAL,
            row["selected_score"],
            timeframe=args.timeframe,
        )
        top3_curves[item["strategy_name"]] = equity_export_oos
        top3_yearly[item["strategy_name"]] = yearly_stats_oos

    try:
        plot_overlay_equity(top3_curves, output_root / "equity_curves_overlay_top3.png", "Top3 OOS equity")
        plot_overlay_drawdown(top3_curves, output_root / "drawdown_overlay_top3.png", "Top3 OOS drawdown")
        plot_overlay_yearly(top3_yearly, output_root / "yearly_pnl_overlay_top3.png", "Top3 OOS yearly pnl_r")
    except Exception as e:
        print(f"Error graficando: {e}")

    # Collect rejection stats for main report
    rejection_log = []
    for item in evaluations:
        status = item.get("status", "N/A")
        reason = item.get("rejection_reason", "N/A")
        rejection_log.append({ "strategy_name": item["strategy_name"], "status": status, "reason": reason })
    pd.DataFrame(rejection_log).to_csv(output_root / "rejection_summary_log.csv", index=False)

    from research_lab.report import export_root_tables
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
