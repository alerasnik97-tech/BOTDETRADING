"""
Campaign 3 Runner
Runner específico para EURUSD Campaign 3 con news dataset correcto.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from research_lab.config import (
    canonical_news_config,
    EngineConfig,
    INITIAL_CAPITAL,
    DEFAULT_DATA_DIRS,
)
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.strategies import STRATEGY_REGISTRY

# Campaign 3 strategies
CAMPAIGN3_STRATEGIES = [
    "campaign3_extended_session_sweep",
    "campaign3_midday_daily_reclaim",
    "campaign3_post_news_continuation",
    "campaign3_afternoon_compression_breakout",
    "campaign3_london_ny_hybrid",
    "campaign3_late_session_momentum",
    "campaign3_mtf_alignment",
]

# Splits
DEV_START = "2020-01-01"
DEV_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2025-12-31"


def run_strategy(strategy_name: str, start_date: str, end_date: str, mode: str = "normal"):
    """Ejecuta una estrategia específica."""
    print(f"\n{'='*60}")
    print(f"  EJECUTANDO: {strategy_name}")
    print(f"  Periodo: {start_date} - {end_date}")
    print(f"  Modo: {mode}")
    print(f"{'='*60}")
    
    if strategy_name not in STRATEGY_REGISTRY:
        print(f"ERROR: Estrategia '{strategy_name}' no encontrada en registro.")
        return None
    
    strategy_module = STRATEGY_REGISTRY[strategy_name]
    
    # Configuración de news con dataset correcto
    news_config = canonical_news_config(
        "EURUSD",
        enabled=True,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
    )
    
    # Configuración del engine
    engine_config = EngineConfig(
        pair="EURUSD",
        risk_pct=0.5,
        execution_mode=mode,
        cost_profile="auto",
        intrabar_policy="auto",
    )
    
    try:
        # Cargar datos bundle
        print("Cargando datos...")
        data_bundle = load_backtest_data_bundle(
            pair="EURUSD",
            data_dirs=DEFAULT_DATA_DIRS,
            start=start_date,
            end=end_date,
            execution_mode=mode,
            target_timeframe=strategy_module.EXPLICIT_TIMEFRAME if hasattr(strategy_module, 'EXPLICIT_TIMEFRAME') else "M15",
        )
        
        if data_bundle is None or data_bundle.frame is None or data_bundle.frame.empty:
            print("ERROR: No se pudieron cargar datos.")
            return None
        
        print(f"Datos cargados: {len(data_bundle.frame)} velas")
        
        # Ejecutar backtest usando run_backtest
        print("Ejecutando backtest...")
        results = run_backtest(
            strategy_module=strategy_module,
            frame=data_bundle.frame,
            engine_config=engine_config,
            news_config=news_config,
            initial_capital=INITIAL_CAPITAL,
            fixed_params=strategy_module.default_params(),
        )
        
        if results is None:
            print("ERROR: Backtest falló.")
            return None
        
        # Calcular métricas
        trades = results["trades"]
        n_trades = len(trades)
        
        if n_trades == 0:
            print("WARNING: No trades generados.")
            return {
                "strategy": strategy_name,
                "period": f"{start_date}-{end_date}",
                "n_trades": 0,
                "pf": None,
                "expectancy": None,
                "max_dd": None,
                "win_rate": None,
                "trades_per_month": 0.0,
            }
        
        # Calcular métricas básicas
        gross_profit = trades[trades["pnl_r"] > 0]["pnl_r"].sum()
        gross_loss = abs(trades[trades["pnl_r"] < 0]["pnl_r"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        expectancy = trades["pnl_r"].mean()
        max_dd = results["equity_curve"].max_drawdown()
        
        wins = len(trades[trades["pnl_r"] > 0])
        win_rate = wins / n_trades if n_trades > 0 else 0
        
        # Calcular trades por mes
        months = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 30.44
        trades_per_month = n_trades / months if months > 0 else 0
        
        result_summary = {
            "strategy": strategy_name,
            "period": f"{start_date}-{end_date}",
            "n_trades": n_trades,
            "pf": pf,
            "expectancy": expectancy,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "trades_per_month": trades_per_month,
        }
        
        print(f"\nResultados:")
        print(f"  Trades: {n_trades}")
        print(f"  PF: {pf:.2f}")
        print(f"  Expectancy: {expectancy:.3f}R")
        print(f"  Max DD: {max_dd:.2f}R")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Trades/mes: {trades_per_month:.1f}")
        
        return result_summary
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_development():
    """Ejecuta development (2020-2023) con todas las estrategias."""
    print("\n" + "="*60)
    print("  CAMPAIGN 3: DEVELOPMENT (2020-2023)")
    print("="*60)
    
    results = []
    
    for strategy_name in CAMPAIGN3_STRATEGIES:
        result = run_strategy(strategy_name, DEV_START, DEV_END, mode="normal_mode")
        if result:
            results.append(result)
    
    # Crear resumen
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("  RESUMEN DEVELOPMENT")
        print("="*60)
        print(df.to_string(index=False))
        
        # Guardar resumen
        output_file = project_root / "results" / "campaign3_development_summary.csv"
        output_file.parent.mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResumen guardado en: {output_file}")
    
    return results


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Campaign 3 Runner")
    parser.add_argument("--phase", choices=["dev", "val", "holdout"], default="dev",
                        help="Fase a ejecutar")
    parser.add_argument("--strategy", help="Estrategia específica (opcional)")
    parser.add_argument("--mode", choices=["normal", "stress", "precision"], default="normal",
                        help="Modo de ejecución")
    
    args = parser.parse_args()
    
    # Mapear modo
    mode_map = {
        "normal": "normal_mode",
        "stress": "conservative_mode",
        "precision": "high_precision_mode",
    }
    execution_mode = mode_map[args.mode]
    
    if args.strategy:
        # Ejecutar estrategia específica
        if args.phase == "dev":
            run_strategy(args.strategy, DEV_START, DEV_END, execution_mode)
        elif args.phase == "val":
            run_strategy(args.strategy, VAL_START, VAL_END, execution_mode)
        elif args.phase == "holdout":
            run_strategy(args.strategy, HOLDOUT_START, HOLDOUT_END, execution_mode)
    else:
        # Ejecutar todas las estrategias de la fase
        if args.phase == "dev":
            run_development()
        elif args.phase == "val":
            print("Validation phase: implementar")
        elif args.phase == "holdout":
            print("Holdout phase: implementar")


if __name__ == "__main__":
    main()
