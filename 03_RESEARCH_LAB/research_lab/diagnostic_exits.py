import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import research_lab.eurusd_am_post_news_external_liquidity_shift_runner as runner

def diagnostic():
    print("--- Running Differential Exit Diagnostic ---")
    news_config = runner.build_news_config()
    news_result = runner.require_operational_news(runner.PAIR, news_config)
    
    start, end = runner.PERIODS["development_2020_2023"]
    filtered_raw = runner._filtered_high_precision_package(start, end)
    m3_raw, m5_raw = runner._build_m3_m5_frames(filtered_raw["mid_m1"])
    
    # Test 1: TP 2.1
    params1 = runner.strategy_module.default_params()
    params1["target_rr"] = 2.1
    params1["break_even_at_r"] = 999.0
    
    # Test 2: TP 5.0
    params2 = runner.strategy_module.default_params()
    params2["target_rr"] = 5.0
    params2["break_even_at_r"] = 999.0

    print("Running Backtest 1 (TP 2.1)...")
    ann1, prec1, sig1 = runner.build_research_frame(start, end, news_events=news_result.events, news_config=news_config)
    res1 = runner.evaluate_period(frame=ann1, precision_package=prec1, signal_log=sig1, params=params1, 
                                engine_config=runner.build_engine_config(), news_result=news_result, 
                                news_config=news_config, start=start, end=end)
    
    print("Running Backtest 2 (TP 5.0)...")
    res2 = runner.evaluate_period(frame=ann1, precision_package=prec1, signal_log=sig1, params=params2, 
                                engine_config=runner.build_engine_config(), news_result=news_result, 
                                news_config=news_config, start=start, end=end)
    
    p1 = res1["summary"]["expectancy_r"]
    p2 = res2["summary"]["expectancy_r"]
    
    print(f"\nExpectancy 1 (TP 2.1): {p1}")
    print(f"Expectancy 2 (TP 5.0): {p2}")
    
    if p1 == p2:
        print("\n[CRITICAL ERROR] Results are identical! Strategy parameters are NOT being applied.")
    else:
        print("\n[SUCCESS] Results differ. Exit parameters are working.")

if __name__ == "__main__":
    diagnostic()
