import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.append(str(PROJECT_ROOT))

from research_lab.eurusd_ltf_objective_entry_replacement_ecb_autopilot import (
    build_research_frame,
    build_engine_config,
    build_news_config,
    evaluate_period,
    AutopilotPaths,
    RESULTS_ROOT,
    STAGE2_RESULTS_DIR,
    FULL_RESULTS_DIR,
    CHECKPOINTS_DIR,
    FAILURE_REPORTS_DIR,
    STATUS_PATH,
    HEARTBEAT_PATH,
    RUNBOOK_PATH,
    STAGE2_DECISION_PATH,
    FINAL_DECISION_PATH,
    FULL_OOS_PATH,
    PRECHECK_AUDIT_PATH,
)
from research_lab.news_filter import require_operational_news
from research_lab.config import with_execution_mode

def run_revalidation():
    print("Iniciando revalidación limpia de ECB_REACTION_LT15M (Sample: 2021)...")
    
    # 1. Build paths
    paths = AutopilotPaths(
        results_root=RESULTS_ROOT,
        stage2_dir=STAGE2_RESULTS_DIR,
        full_dir=FULL_RESULTS_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        failure_reports_dir=FAILURE_REPORTS_DIR,
        status_path=STATUS_PATH,
        heartbeat_path=HEARTBEAT_PATH,
        runbook_path=RUNBOOK_PATH,
        stage2_decision_path=STAGE2_DECISION_PATH,
        final_decision_path=FINAL_DECISION_PATH,
        full_oos_path=FULL_OOS_PATH,
        precheck_audit_path=PRECHECK_AUDIT_PATH,
    )
    
    # 2. Build news config
    news_config = build_news_config()
    news_result = require_operational_news("EURUSD", news_config)
    
    # 3. Load data for 2021
    print("Cargando datos y generando señales para 2021...")
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    frame, precision_package, signal_log = build_research_frame(start_date, end_date)
    
    # 4. Apply delay < 15m filter to signal_log
    # signal_log columns: signal_time, sweep_time_ny, extreme_time_ny, etc.
    # We need to calculate delay in minutes.
    # sweep_time_ny is the H1 signal time. 
    # extreme_time_ny is the timestamp of the M3 extreme candle.
    # The trigger happens when M3 price breaks the extreme candle.
    # Wait, build_research_frame already annotated the frame with ecb_signal.
    # We need to filter the frame itself or the signals before backtest.
    
    # Let's calculate delay for each signal in signal_log
    signal_log['extreme_dt'] = pd.to_datetime(signal_log['extreme_time_ny'], utc=True).dt.tz_convert("America/New_York")
    signal_log['signal_dt'] = pd.to_datetime(signal_log['signal_time'], utc=True).dt.tz_convert("America/New_York")
    
    # Delay = Time between extreme and the signal (which is at the end of the H1 candle where trigger occurs)
    # Actually, in the autopilot logic, the trigger occurs during the H1 candle.
    # Let's re-read annotate_ecb_frame:
    # m3_window = result.loc[(result.index > sweep_ts - pd.Timedelta(hours=1)) & (result.index <= sweep_ts)].copy()
    # extreme = _extreme_candle(m3_window, ...)
    # The signal_time is sweep_ts (the end of the H1 candle).
    # The extreme timestamp is the M3 bar within that hour.
    
    signal_log['delay_min'] = (signal_log['signal_dt'] - signal_log['extreme_dt']).dt.total_seconds() / 60
    
    print(f"Señales ECB originales en 2021: {len(signal_log)}")
    fast_signals = signal_log[signal_log['delay_min'] <= 15].copy()
    print(f"Señales ECB_REACTION_LT15M en 2021: {len(fast_signals)}")
    
    if len(fast_signals) == 0:
        print("ERROR: No hay señales que cumplan el criterio de delay en 2021.")
        return

    # 5. Run backtests
    params = {"session_name": "all_day", "target_rr": 1.5}
    
    # BASELINE: slippage 0.1
    print("\nEjecutando Baseline (Slippage 0.1)...")
    config_baseline = build_engine_config(execution_mode="high_precision_mode")
    # evaluate_period internally uses the frame and signal_log provided.
    # We must pass the FILTERED signal_log.
    res_baseline = evaluate_period(
        frame=frame,
        precision_package=precision_package,
        signal_log=fast_signals,
        params=params,
        engine_config=config_baseline,
        news_result=news_result,
        news_config=news_config,
        start=start_date,
        end=end_date
    )
    
    # STRESS: slippage 0.4
    print("\nEjecutando Stress (Slippage 0.4)...")
    config_stress = build_engine_config(execution_mode="high_precision_mode")
    # Override slippage
    from dataclasses import replace
    config_stress = replace(config_stress, slippage_pips=0.4)
    
    res_stress = evaluate_period(
        frame=frame,
        precision_package=precision_package,
        signal_log=fast_signals,
        params=params,
        engine_config=config_stress,
        news_result=news_result,
        news_config=news_config,
        start=start_date,
        end=end_date
    )
    
    # 6. Report results
    def print_metrics(res, label):
        s = res['summary']
        print(f"--- {label} ---")
        print(f"N: {s['total_trades']}")
        print(f"PF: {res['summary'].get('profit_factor', s.get('profit_factor')):.3f}")
        print(f"Exp: {s['expectancy_r']:.3f}R")
        print(f"DD: {s['max_drawdown_r_closed_trades']:.3f}R")
        print(f"WR: {s['win_rate']:.1f}%")

    print_metrics(res_baseline, "RESULTADOS BASELINE 2021")
    print_metrics(res_stress, "RESULTADOS STRESS 2021")

if __name__ == "__main__":
    run_revalidation()
