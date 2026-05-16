#!/usr/bin/env python3
"""
Smoke test end-to-end para Campaign 3B.
Verifica que la integración técnica del motor es correcta antes de abrir hipótesis nuevas.
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

# Usar estrategia existente como prueba (ya funciona)
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module

def main():
    print("=" * 60)
    print("CAMPAIGN 3B - SMOKE TEST ENGINE INTEGRATION")
    print("=" * 60)
    
    # Configuración de prueba
    PAIR = "EURUSD"
    START = "2020-01-01"
    END = "2020-12-31"  # Solo 1 año para smoke test rápido
    EXECUTION_MODE = "normal_mode"
    TARGET_TIMEFRAME = strategy_module.EXPLICIT_TIMEFRAME
    
    print(f"\n[1] Cargando datos ({PAIR}, {START} - {END}, {EXECUTION_MODE}, {TARGET_TIMEFRAME})...")
    try:
        data_bundle = load_backtest_data_bundle(
            pair=PAIR,
            data_dirs=DEFAULT_DATA_DIRS,
            start=START,
            end=END,
            execution_mode=EXECUTION_MODE,
            target_timeframe=TARGET_TIMEFRAME,
        )
        frame = data_bundle.frame
        print(f"[OK] Frame cargado: {len(frame)} bares")
        print(f"   Data source: {data_bundle.data_source_used}")
        print(f"   Precision package: {'Yes' if data_bundle.precision_package else 'No'}")
    except Exception as e:
        print(f"[ERROR] ERROR cargando datos: {e}")
        return False
    
    print(f"\n[2] Verificando columnas requeridas...")
    required_columns = ["open", "high", "low", "close", "atr14", "ema20", "ema50", "prev_day_high", "prev_day_low"]
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        print(f"[ERROR] Columnas faltantes: {missing}")
        return False
    print(f"[OK] Columnas requeridas presentes")
    
    print(f"\n[3] Cargando News Fortress...")
    try:
        news_config = canonical_news_config(PAIR, enabled=True)
        news_result = require_operational_news(PAIR, news_config)
        print(f"[OK] News cargadas: {len(news_result.events)} eventos")
        print(f"   Enabled: {news_result.enabled}")
    except Exception as e:
        print(f"[ERROR] ERROR cargando news: {e}")
        return False
    
    print(f"\n[4] Construyendo news_block...")
    try:
        bar_open_index = entry_open_index(frame.index)
        news_block = build_entry_block(bar_open_index, news_result.events, news_config)
        print(f"[OK] news_block construido: {len(news_block)} bares")
        print(f"   Bloqueados: {news_block.sum()} bares")
    except Exception as e:
        print(f"[ERROR] ERROR construyendo news_block: {e}")
        return False
    
    print(f"\n[5] Configurando engine...")
    try:
        engine_config = EngineConfig(
            pair=PAIR,
            risk_pct=0.5,
            execution_mode=EXECUTION_MODE,
            max_trades_per_day=2,
            session_cutoff=None,
        )
        print(f"[OK] Engine config creado")
    except Exception as e:
        print(f"[ERROR] ERROR configurando engine: {e}")
        return False
    
    print(f"\n[6] Ejecutando backtest...")
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
        print(f"[ERROR] ERROR ejecutando backtest: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[7] Analizando resultados...")
    try:
        summary, trades, monthly, yearly, equity = summarize_result(
            strategy_module.NAME,
            result.trades,
            result.equity_curve,
            result.params,
            news_result.enabled,
            100000,
            None,
            timeframe=TARGET_TIMEFRAME,
        )
        print(f"[OK] Resultados analizados")
        print(f"   PF: {summary.get('profit_factor', 'N/A')}")
        print(f"   Expectancy: {summary.get('expectancy_r', 'N/A')}")
        print(f"   Win Rate: {summary.get('win_rate', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] ERROR analizando resultados: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED [OK]")
    print("Integración técnica del motor verificada correctamente.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
