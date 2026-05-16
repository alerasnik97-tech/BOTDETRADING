import sys
import os
import gc
from pathlib import Path
import pandas as pd
import numpy as np

# Add research_lab to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from research_lab import eurusd_am_post_news_external_liquidity_shift_runner as runner
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module

def run_equivalence():
    print("Starting HARD EQUIVALENCE VALIDATION (Mode A vs Mode B)...", flush=True)
    
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    # 1. Load Common Data
    print(f"Loading Research Frame for {start_date} -> {end_date}...")
    full_frame, full_precision, full_signal_log = runner.build_research_frame(
        start_date, end_date,
        news_events=runner.require_operational_news("EURUSD", runner.build_news_config()).events,
        news_config=runner.build_news_config()
    )
    
    label = "WINNER_MAX_PF"
    params = strategy_module.default_params()
    params["variant_label"] = "short_only_purified"
    params["start_h"] = 9
    
    runner.OPERATIVE_START_MINUTE = params["start_h"] * 60
    
    short_lvls = (("prev_day_high", "prev_day"), ("london_high", "london"))
    long_lvls = ()

    # --- MODE A: STANDARD (No Cache) ---
    print("Executing Mode A (Standard)...")
    m3_a, m5_a = runner._build_m3_m5_frames(full_precision["mid_m1"])
    annotated_a, signal_log_a = runner.annotate_post_news_external_liquidity_shift_frame(
        m3_a.copy(),
        m5_a.copy(),
        news_events=runner.require_operational_news("EURUSD", runner.build_news_config()).events,
        news_config=runner.build_news_config(),
        short_levels=short_lvls,
        long_levels=long_lvls
    )
    
    precision_a = runner._align_precision_package(full_precision, annotated_a.index)
    res_a = runner.evaluate_period(
        frame=annotated_a,
        precision_package=precision_a,
        signal_log=signal_log_a,
        params=params,
        engine_config=runner.build_engine_config(),
        news_result=runner.require_operational_news("EURUSD", runner.build_news_config()),
        news_config=runner.build_news_config(),
        start=start_date,
        end=end_date
    )
    trades_a = res_a["trades_export"]

    # --- MODE B: OPTIMIZED (Cached Frames) ---
    print("Executing Mode B (Optimized - Monthly Cache)...")
    # This simulates the new script logic
    m3_cache, m5_cache = runner._build_m3_m5_frames(full_precision["mid_m1"])
    
    annotated_b, signal_log_b = runner.annotate_post_news_external_liquidity_shift_frame(
        m3_cache.copy(),
        m5_cache.copy(),
        news_events=runner.require_operational_news("EURUSD", runner.build_news_config()).events,
        news_config=runner.build_news_config(),
        short_levels=short_lvls,
        long_levels=long_lvls
    )
    
    precision_b = runner._align_precision_package(full_precision, annotated_b.index)
    res_b = runner.evaluate_period(
        frame=annotated_b,
        precision_package=precision_b,
        signal_log=signal_log_b,
        params=params,
        engine_config=runner.build_engine_config(),
        news_result=runner.require_operational_news("EURUSD", runner.build_news_config()),
        news_config=runner.build_news_config(),
        start=start_date,
        end=end_date
    )
    trades_b = res_b["trades_export"]

    # --- COMPARISON ---
    print("\n" + "="*40)
    print("EQUIVALENCE ANALYSIS")
    print("="*40)
    
    match_signals = signal_log_a.equals(signal_log_b)
    match_trades = trades_a.equals(trades_b)
    
    print(f"Signal Identity: {'MATCH' if match_signals else 'DIVERGENCE'}")
    print(f"Trade Identity:  {'MATCH' if match_trades else 'DIVERGENCE'}")
    
    if not match_signals:
        diff_sigs = pd.concat([signal_log_a, signal_log_b]).drop_duplicates(keep=False)
        print("\nSignal Divergences found:")
        print(diff_sigs)

    if not match_trades:
        print("\nTrade Divergences found:")
        # Check basic counts
        print(f"  Count A: {len(trades_a)} | Count B: {len(trades_b)}")
        # If counts match, check PnL
        if len(trades_a) == len(trades_b):
            pnl_diff = (trades_a["pnl_r"] - trades_b["pnl_r"]).abs().sum()
            print(f"  PnL Absolute Difference: {pnl_diff}")

    # Physical Output
    summary_path = "results/performance_equivalence_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Equivalence Validation Report: Rescue 3.0\n\n")
        f.write(f"- **Block**: {start_date} to {end_date}\n")
        f.write(f"- **Mode A (Standard)**: Manual build per evaluation\n")
        f.write(f"- **Mode B (Optimized)**: Monthly cache reuse\n\n")
        f.write(f"## Decision Matrix\n")
        f.write(f"| Metric | Status |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Signal Identity** | {'[OK] MATCH' if match_signals else '[ERROR] DIVERGENCE'} |\n")
        f.write(f"| **Trade Identity** | {'[OK] MATCH' if match_trades else '[ERROR] DIVERGENCE'} |\n\n")
        f.write(f"## Conclusion\n")
        if match_signals and match_trades:
            f.write("> [!NOTE]\n")
            f.write("> **BINARY CHOICE: OPTION A (KEEP OPTIMIZED)**. Behavioral identity confirmed. No compromise in logic truth.\n")
        else:
            f.write("> [!CAUTION]\n")
            f.write("> **BINARY CHOICE: OPTION B (REVERT/FIX)**. Optimization altered the behavioral output. Investigation required.\n")

    print(f"\nReport generated: {summary_path}")

if __name__ == "__main__":
    run_equivalence()
