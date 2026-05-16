from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import json
import itertools
import gc
from typing import Any

# Add project root to path
PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import research_lab.eurusd_am_post_news_external_liquidity_shift_runner as runner
from research_lab.config import EngineConfig, NY_TZ

def run_audit():
    print("--- Starting High-Efficiency Audit Grid ---")
    news_config = runner.build_news_config()
    news_result = runner.require_operational_news(runner.PAIR, news_config)
    
    print("Pre-loading Price Data (this may take 2-3 minutes)...")
    # Pre-load the high precision package once to reuse it
    filtered_raw = runner._filtered_high_precision_package(*runner.PERIODS["development_2020_2023"])
    m3_frame_raw, m5_frame_raw = runner._build_m3_m5_frames(filtered_raw["mid_m1"])
    
    variants = [
        ("all_sources", runner.DEFAULT_SHORT_LEVELS, runner.DEFAULT_LONG_LEVELS),
        ("purified (no asia)", (("prev_day_high", "prev_day"), ("london_high", "london")), (("prev_day_low", "prev_day"), ("london_low", "london"))),
        ("short_only_purified", (("prev_day_high", "prev_day"), ("london_high", "london")), ()),
    ]
    hours = [7, 8, 9]
    bes = [None, 1.2]
    tps = [2.1, 2.4, 3.0]
    
    csv_path = "robustness_audit_grid_dev.csv"
    
    # Initialize CSV with headers
    pd.DataFrame(columns=[
        "variant", "start_h", "be", "tp", "trades", "pf", "expectancy", "pnl"
    ]).to_csv(csv_path, index=False)
    
    for v_name, s_lvls, l_lvls in variants:
        for h in hours:
            print(f"--- Processing Segment: {v_name} @ {h:02d}:00 ---")
            
            # Annotate and build signal log for this specific hour
            runner.OPERATIVE_START_MINUTE = h * 60
            annotated, signal_log = runner.annotate_post_news_external_liquidity_shift_frame(
                m3_frame_raw.copy(),
                m5_frame_raw.copy(),
                news_events=news_result.events,
                news_config=news_config,
                short_levels=s_lvls,
                long_levels=l_lvls,
            )
            
            if signal_log.empty:
                print(f"    - No signals for {v_name} @ {h}")
                continue

            precision_package = runner._align_precision_package(filtered_raw, annotated.index)
            
            for be, tp in itertools.product(bes, tps):
                params = runner.strategy_module.default_params()
                params["target_rr"] = tp
                params["break_even_at_r"] = be if be is not None else 999.0
                
                print(f"    > Running: BE={be}, TP={tp}...", end="", flush=True)
                
                eval_result = runner.evaluate_period(
                    frame=annotated,
                    precision_package=precision_package,
                    signal_log=signal_log,
                    params=params,
                    engine_config=runner.build_engine_config(),
                    news_result=news_result,
                    news_config=news_config,
                    start=runner.PERIODS["development_2020_2023"][0],
                    end=runner.PERIODS["development_2020_2023"][1],
                )
                
                summary = eval_result["summary"]
                row = {
                    "variant": v_name,
                    "start_h": h,
                    "be": be if be is not None else "",
                    "tp": tp,
                    "trades": summary["total_trades"],
                    "pf": round(summary.get("profit_factor", 0), 3),
                    "expectancy": round(summary.get("expectancy_r", 0), 4),
                    "pnl": round(float(summary.get("expectancy_r", 0) * summary.get("total_trades", 0)), 2)
                }
                
                # Incremental Save
                pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)
                print(f" Done. Expectancy: {row['expectancy']}")
                
            # Memory Cleanup
            del annotated, signal_log, precision_package
            gc.collect()

    print(f"\n--- Audit Grid Phase 2 Complete. Results persisted in {csv_path} ---")

if __name__ == "__main__":
    run_audit()
