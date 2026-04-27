#!/usr/bin/env python3
"""
Runner principal para Campaign 3B.
Ejecuta las 4 estrategias en 3 splits (Development, Validation, Holdout) con 3 niveles de fricción.
"""

from pathlib import Path
import sys
import pandas as pd
from datetime import timedelta

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CURRENT_DIR))

from research_lab.config import EngineConfig, canonical_news_config, DEFAULT_DATA_DIRS, with_execution_mode
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest, entry_open_index
from research_lab.news_filter import require_operational_news, build_entry_block
from research_lab.report import summarize_result

# Estrategias Campaign 3B
STRATEGIES = [
    "campaign3b_midday_reclaim",
    "campaign3b_compression_breakout",
    "campaign3b_post_news_continuation",
    "campaign3b_session_expansion",
]

# Splits de validación
SPLITS = {
    "development": ("2020-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2024-12-31"),
    "holdout": ("2025-01-01", "2025-12-31"),
}

# Niveles de fricción
FRICTION_LEVELS = {
    "baseline": {"execution_mode": "normal_mode", "spread": 1.2, "slippage": 0.2},
    "medium": {"execution_mode": "conservative_mode", "spread": 1.5, "slippage": 0.3},
    "hard": {"execution_mode": "conservative_mode", "spread": 2.0, "slippage": 0.5},
}

def load_strategy_module(strategy_name: str):
    """Carga dinámicamente el módulo de estrategia."""
    from research_lab.strategies import STRATEGY_REGISTRY
    return STRATEGY_REGISTRY[strategy_name]

def run_single_backtest(strategy_name: str, split: str, friction: str) -> dict:
    """Ejecuta un backtest individual."""
    strategy_module = load_strategy_module(strategy_name)
    start, end = SPLITS[split]
    friction_config = FRICTION_LEVELS[friction]
    
    print(f"\n  [{strategy_name}] {split} {friction}...")
    
    # Cargar datos
    data_bundle = load_backtest_data_bundle(
        pair="EURUSD",
        data_dirs=DEFAULT_DATA_DIRS,
        start=start,
        end=end,
        execution_mode=friction_config["execution_mode"],
        target_timeframe=strategy_module.EXPLICIT_TIMEFRAME,
    )
    frame = data_bundle.frame
    
    # Cargar noticias
    news_config = canonical_news_config("EURUSD", enabled=True)
    news_result = require_operational_news("EURUSD", news_config)
    
    # Construir news_block
    bar_open_index = entry_open_index(frame.index)
    news_block = build_entry_block(bar_open_index, news_result.events, news_config)
    
    # Configurar engine
    engine_config = EngineConfig(
        pair="EURUSD",
        risk_pct=0.5,
        execution_mode=friction_config["execution_mode"],
        max_trades_per_day=2,
        session_cutoff="19:00",  # Cierre a las 19:00 NY
    )
    
    # Ejecutar backtest
    result = run_backtest(
        strategy_module=strategy_module,
        frame=frame,
        params=strategy_module.default_params(),
        engine_config=engine_config,
        news_block=news_block,
        news_filter_used=news_result.enabled,
        precision_package=data_bundle.precision_package,
        data_source_used=data_bundle.data_source_used,
        news_events=news_result.events,
        news_settings=news_config,
    )
    
    # Analizar resultados
    summary, trades, monthly, yearly, equity = summarize_result(
        strategy_module.NAME,
        result.trades,
        result.equity_curve,
        result.params,
        news_result.enabled,
        100000,
        None,
        timeframe=strategy_module.EXPLICIT_TIMEFRAME,
    )
    
    return {
        "strategy": strategy_name,
        "split": split,
        "friction": friction,
        "n_trades": len(result.trades),
        "pf": summary.get("profit_factor", 0),
        "expectancy_r": summary.get("expectancy_r", 0),
        "drawdown_r": summary.get("max_drawdown_r", 0),
        "win_rate": summary.get("win_rate", 0),
        "total_pnl_r": summary.get("total_pnl_r", 0),
    }

def main():
    print("=" * 70)
    print("CAMPAIGN 3B - MAIN RUNNER")
    print("=" * 70)
    
    results = []
    
    for strategy_name in STRATEGIES:
        print(f"\n=== Estrategia: {strategy_name} ===")
        
        for split in SPLITS.keys():
            for friction in FRICTION_LEVELS.keys():
                try:
                    result = run_single_backtest(strategy_name, split, friction)
                    results.append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "strategy": strategy_name,
                        "split": split,
                        "friction": friction,
                        "n_trades": 0,
                        "pf": 0,
                        "expectancy_r": 0,
                        "drawdown_r": 0,
                        "win_rate": 0,
                        "total_pnl_r": 0,
                        "error": str(e),
                    })
    
    # Exportar resultados
    df_results = pd.DataFrame(results)
    
    # Crear directorio de resultados
    results_dir = Path("results/campaign3b")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar CSV completo
    df_results.to_csv(results_dir / "campaign3b_full_results.csv", index=False)
    
    # Guardar resumen por split
    for split in SPLITS.keys():
        df_split = df_results[df_results["split"] == split].copy()
        df_split.to_csv(results_dir / f"campaign3b_{split}_results.csv", index=False)
    
    print("\n" + "=" * 70)
    print("RESULTADOS RESUMIDOS")
    print("=" * 70)
    
    # Resumen development baseline
    dev_baseline = df_results[(df_results["split"] == "development") & (df_results["friction"] == "baseline")]
    print("\n[Development Baseline]")
    for _, row in dev_baseline.iterrows():
        print(f"  {row['strategy']}: N={row['n_trades']}, PF={row['pf']:.2f}, Exp={row['expectancy_r']:.3f}, DD={row['drawdown_r']:.2f}")
    
    # Resumen validation baseline
    val_baseline = df_results[(df_results["split"] == "validation") & (df_results["friction"] == "baseline")]
    print("\n[Validation Baseline]")
    for _, row in val_baseline.iterrows():
        print(f"  {row['strategy']}: N={row['n_trades']}, PF={row['pf']:.2f}, Exp={row['expectancy_r']:.3f}, DD={row['drawdown_r']:.2f}")
    
    # Resumen holdout baseline
    holdout_baseline = df_results[(df_results["split"] == "holdout") & (df_results["friction"] == "baseline")]
    print("\n[Holdout Baseline]")
    for _, row in holdout_baseline.iterrows():
        print(f"  {row['strategy']}: N={row['n_trades']}, PF={row['pf']:.2f}, Exp={row['expectancy_r']:.3f}, DD={row['drawdown_r']:.2f}")
    
    # Resumen stress test (hard friction)
    dev_hard = df_results[(df_results["split"] == "development") & (df_results["friction"] == "hard")]
    print("\n[Development Hard Friction]")
    for _, row in dev_hard.iterrows():
        print(f"  {row['strategy']}: N={row['n_trades']}, PF={row['pf']:.2f}, Exp={row['expectancy_r']:.3f}, DD={row['drawdown_r']:.2f}")
    
    print(f"\nResultados guardados en: {results_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
