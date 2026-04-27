from pathlib import Path
import pandas as pd
import numpy as np
from research_lab.data_loader import load_backtest_data_bundle
from research_lab.config import DEFAULT_PAIR, DEFAULT_DATA_DIRS, EngineConfig, NewsConfig, with_execution_mode
from research_lab.strategies.adr_exhaustion_fade import signal as adr_signal
from research_lab.engine import run_backtest
from research_lab.news_filter import load_news_events, build_entry_block

def debug_engine():
    bundle = load_backtest_data_bundle(
        DEFAULT_PAIR,
        [Path(d) for d in DEFAULT_DATA_DIRS],
        "2024-01-01",
        "2024-02-01",
        "conservative_mode"
    )
    df = bundle.frame
    strategy_module = type('obj', (object,), {'NAME': 'adr_exhaustion_fade', 'WARMUP_BARS': 200, 'signal': adr_signal})
    
    params = {"target_rr": 1.5, "break_even_at_r": None, "exhaustion_mult": 1.5, "session_name": "light_fixed"}
    engine_config = with_execution_mode(EngineConfig(), "conservative_mode")
    news_config = NewsConfig(enabled=True)
    
    news_res = load_news_events(DEFAULT_PAIR, news_config)
    news_block = build_entry_block(df.index, news_res.events, news_config)
    
    res = run_backtest(
        strategy_module,
        df,
        params,
        engine_config,
        news_block,
        news_res.enabled
    )
    
    print(f"Total Trades: {len(res.trades)}")
    if len(res.trades) == 0:
        # Investigar por que
        for i in range(200, len(df)):
            sig = adr_signal(df, i, params)
            if sig:
                print(f"Signal at {df.index[i]}")
                # Simular chequeo de engine
                # entry_spread_pips estimate
                # fill_allowed
                # news_block
                # shock_candle
                print(f"  News Block: {news_block[i+1] if i+1 < len(df) else 'OOB'}")
                print(f"  Range ATR: {df['range_atr'].iat[i]:.2f} (max {engine_config.shock_candle_atr_max})")

if __name__ == "__main__":
    debug_engine()
