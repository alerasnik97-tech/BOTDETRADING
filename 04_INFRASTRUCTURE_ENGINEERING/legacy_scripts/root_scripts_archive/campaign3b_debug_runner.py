#!/usr/bin/env python3
"""
Debug runner para Campaign 3B.
Testea una sola estrategia en un periodo corto para identificar errores.
"""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CURRENT_DIR))

from research_lab.config import EngineConfig, canonical_news_config, DEFAULT_DATA_DIRS
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest, entry_open_index
from research_lab.news_filter import require_operational_news, build_entry_block
from research_lab.report import summarize_result

def main():
    print("=" * 60)
    print("CAMPAIGN 3B - DEBUG RUNNER")
    print("=" * 60)
    
    strategy_name = "campaign3b_midday_reclaim"
    start = "2020-01-01"
    end = "2020-02-01"  # Solo 1 mes para debug
    
    print(f"\nTesteando: {strategy_name}")
    print(f"Periodo: {start} - {end}")
    
    # Cargar estrategia
    from research_lab.strategies import STRATEGY_REGISTRY
    strategy_module = STRATEGY_REGISTRY[strategy_name]
    
    print(f"Timeframe: {strategy_module.EXPLICIT_TIMEFRAME}")
    print(f"Warmup bars: {strategy_module.WARMUP_BARS}")
    print(f"Default params: {strategy_module.default_params()}")
    
    # Cargar datos
    print("\n[1] Cargando datos...")
    try:
        data_bundle = load_backtest_data_bundle(
            pair="EURUSD",
            data_dirs=DEFAULT_DATA_DIRS,
            start=start,
            end=end,
            execution_mode="normal_mode",
            target_timeframe=strategy_module.EXPLICIT_TIMEFRAME,
        )
        frame = data_bundle.frame
        print(f"[OK] Frame cargado: {len(frame)} bares")
    except Exception as e:
        print(f"[ERROR] Cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Cargar noticias
    print("\n[2] Cargando noticias...")
    try:
        news_config = canonical_news_config("EURUSD", enabled=True)
        news_result = require_operational_news("EURUSD", news_config)
        print(f"[OK] News cargadas: {len(news_result.events)} eventos")
    except Exception as e:
        print(f"[ERROR] Cargando news: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Construir news_block
    print("\n[3] Construyendo news_block...")
    try:
        bar_open_index = entry_open_index(frame.index)
        news_block = build_entry_block(bar_open_index, news_result.events, news_config)
        print(f"[OK] news_block construido: {len(news_block)} bares")
    except Exception as e:
        print(f"[ERROR] Construyendo news_block: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Configurar engine
    print("\n[4] Configurando engine...")
    try:
        engine_config = EngineConfig(
            pair="EURUSD",
            risk_pct=0.5,
            execution_mode="normal_mode",
            max_trades_per_day=2,
            session_cutoff="19:00",
        )
        print("[OK] Engine config creado")
    except Exception as e:
        print(f"[ERROR] Configurando engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Ejecutar backtest
    print("\n[5] Ejecutando backtest...")
    try:
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
        print(f"[OK] Backtest ejecutado")
        print(f"   Strategy: {result.strategy_name}")
        print(f"   Trades: {len(result.trades)}")
        print(f"   Equity points: {len(result.equity_curve)}")
    except Exception as e:
        print(f"[ERROR] Ejecutando backtest: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analizar resultados
    print("\n[6] Analizando resultados...")
    try:
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
        print(f"[OK] Resultados analizados")
        print(f"   PF: {summary.get('profit_factor', 'N/A')}")
        print(f"   Expectancy: {summary.get('expectancy_r', 'N/A')}")
        print(f"   Win Rate: {summary.get('win_rate', 'N/A')}")
        print(f"   Total PnL R: {summary.get('total_pnl_r', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Analizando resultados: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("DEBUG TEST PASSED [OK]")
    print("=" * 60)

if __name__ == "__main__":
    main()
