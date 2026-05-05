import pandas as pd
import numpy as np
import os
import json

# Paths
ROOT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
REPORTS_DIR = os.path.join(ROOT_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
LITE_DIR = os.path.join(REPORTS_DIR, "phase50y_lite")
RAW_TRADES_PATH = os.path.join(ROOT_DIR, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")

TARGET_MONTHS = ["2015-01", "2015-10", "2015-11", "2017-05", "2017-08", "2020-04", "2024-10", "2025-02", "2025-11"]

def calibrate_risk():
    print("--- TAREA 1: Inventario ---")
    inventory = [
        {"file_path": "MANIPULANTE/10_LOGS_PAPER/dry_run_decisions", "exists": True, "rows": 0, "columns": "", "contains_real_fills_yes_no": "NO", "contains_spread_yes_no": "NO", "contains_commission_yes_no": "NO", "contains_slippage_yes_no": "NO", "usable_for_cost_calibration_yes_no": "NO", "notes": "Empty"},
        {"file_path": "MANIPULANTE/15_FORWARD_DEMO_SCORECARD/forward_demo_master_scorecard.csv", "exists": True, "rows": 1, "columns": "date,status,trades...", "contains_real_fills_yes_no": "NO", "contains_spread_yes_no": "NO", "contains_commission_yes_no": "NO", "contains_slippage_yes_no": "NO", "usable_for_cost_calibration_yes_no": "NO", "notes": "No trades executed"},
    ]
    pd.DataFrame(inventory).to_csv(os.path.join(ROOT_DIR, "BOT_V2_DAYTIME_LAB", "reports", "PHASE51_EXECUTION_COST_SOURCE_INVENTORY.csv"), index=False)
    
    print("--- TAREA 2: Risk Distribution ---")
    df_raw = pd.read_csv(RAW_TRADES_PATH)
    df_raw['ym'] = pd.to_datetime(df_raw['entry_time'], utc=True).dt.strftime('%Y-%m')
    
    # Filter for Group B months
    df_group_b = df_raw[df_raw['ym'].isin(TARGET_MONTHS)].copy()
    
    # risk_pips = abs(entry - sl) * 10000
    df_group_b['risk_pips'] = (df_group_b['entry_price'] - df_group_b['sl']).abs() * 10000
    
    # Distribution
    dist = {
        "sample": len(df_group_b),
        "mean": df_group_b['risk_pips'].mean(),
        "median": df_group_b['risk_pips'].median(),
        "p10": df_group_b['risk_pips'].quantile(0.1),
        "p25": df_group_b['risk_pips'].quantile(0.25),
        "p75": df_group_b['risk_pips'].quantile(0.75),
        "p90": df_group_b['risk_pips'].quantile(0.9),
        "min": df_group_b['risk_pips'].min(),
        "max": df_group_b['risk_pips'].max()
    }
    
    df_group_b[['ym', 'trade_id', 'type', 'entry_price', 'sl', 'risk_pips']].to_csv(os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_RISK_DISTRIBUTION.csv"), index=False)
    
    with open(os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_RISK_STATS.json"), "w") as f:
        json.dump(dist, f, indent=4)
        
    print(f"Risk stats: {dist}")

if __name__ == "__main__":
    calibrate_risk()
