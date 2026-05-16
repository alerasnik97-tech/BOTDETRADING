from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from research_lab.config import EngineConfig, NY_TZ, NewsConfig, DEFAULT_DATA_DIRS
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.engine import run_backtest
from research_lab.strategies import am_silver_bullet_ny


def main():
    # 1. Configuration
    pair = "EURUSD"
    start_date = "2020-01-01"
    end_date = "2025-04-15"
    
    data_dirs = list(DEFAULT_DATA_DIRS)
    news_events_path = Path("data/news_eurusd_am_fortress_v3.csv")
    
    if not news_events_path.exists():
        print(f"Error: No se encontro el dataset de noticias AM-Grade: {news_events_path}")
        return

    # 2. Engine and News Settings
    engine_config = EngineConfig(
        pair=pair,
        risk_pct=1.0,
        execution_mode="high_precision_mode", # Para mayor fidelidad en los limites
        enforce_hard_stop=True,
    )
    
    news_events = pd.read_csv(news_events_path)
    news_events["timestamp_ny"] = pd.to_datetime(news_events["timestamp_ny"], utc=True).dt.tz_convert(NY_TZ)
    
    # 3. Load Data
    print(f"[{am_silver_bullet_ny.NAME}] Cargando datos para {pair} ({start_date} a {end_date})...")
    bundle = load_backtest_data_bundle(
        pair=pair,
        data_dirs=data_dirs,
        start=start_date,
        end=end_date,
        execution_mode=engine_config.execution_mode,
        target_timeframe="M5",
    )
    
    # 4. Run Backtest
    print(f"[{am_silver_bullet_ny.NAME}] Ejecutando backtest con News Fortress v3 activo...")
    params = am_silver_bullet_ny.default_params()
    
    # El runner de AM debe filtrar noticias (usando el dataset curado)
    news_settings = NewsConfig()
    result = run_backtest(
        strategy_module=am_silver_bullet_ny,
        frame=bundle.frame,
        params=params,
        engine_config=engine_config,
        news_block=[], # No usamos block estatico
        news_filter_used=True,
        precision_package=bundle.precision_package,
        news_events=news_events,
        news_settings=news_settings,
    )
    
    # 5. Results
    trades = result.trades
    if trades.empty:
        print(f"[{am_silver_bullet_ny.NAME}] No se generaron trades en este periodo.")
    else:
        print(f"[{am_silver_bullet_ny.NAME}] Backtest concluido con {len(trades)} trades.")
        pf = trades["pnl_usd"].loc[trades["pnl_usd"] > 0].sum() / abs(trades["pnl_usd"].loc[trades["pnl_usd"] < 0].sum())
        print(f"Profit Factor: {pf:.2f}")
        
    trades_path = Path("research_lab/results/am_silver_bullet_trades.csv")
    trades.to_csv(trades_path, index=False)
    print(f"Resultados guardados en {trades_path}")


if __name__ == "__main__":
    main()
