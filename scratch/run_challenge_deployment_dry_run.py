"""
Challenge Deployment Dry-Run & Statistics
Simulates hundreds of rolling challenges to find the optimal deployment playbook.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
GLOBAL_LEDGER = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
CORE_LEDGER = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
OUTPUT_REPORT = ROOT / "results" / "SCBI_CHALLENGE_DEPLOYMENT_DRY_RUN.json"

INITIAL_BALANCE = 100000
TARGET_PROFIT = 0.10 # 10%
DAILY_LIMIT = 0.05 # 5%
TOTAL_LIMIT = 0.10 # 10%

RISK_LEVELS = [0.25, 0.50, 0.75] # % risk per trade

def simulate_window_challenge(df, start_idx, risk_pct):
    balance = INITIAL_BALANCE
    max_balance = INITIAL_BALANCE
    daily_start_balance = INITIAL_BALANCE
    
    current_date = None
    
    # Process trades from start_idx until pass or fail
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        date = row.get("session_date", row.get("date"))
        pnl_r = row["pnl_r"]
        
        if date != current_date:
            daily_start_balance = balance
            current_date = date
            
        risk_amt = balance * (risk_pct / 100)
        trade_pnl = pnl_r * risk_amt
        balance += trade_pnl
        max_balance = max(max_balance, balance)
        
        # Check Breach
        if (daily_start_balance - balance) / daily_start_balance >= DAILY_LIMIT:
            return "BREACH_DAILY", i - start_idx
        if (INITIAL_BALANCE - balance) / INITIAL_BALANCE >= TOTAL_LIMIT:
            return "BREACH_TOTAL", i - start_idx
            
        # Check Pass
        if (balance - INITIAL_BALANCE) / INITIAL_BALANCE >= TARGET_PROFIT:
            return "PASS", i - start_idx
            
    return "TIMEOUT", len(df) - start_idx

def run_dry_run():
    if not GLOBAL_LEDGER.exists() or not CORE_LEDGER.exists():
        return {"error": "Ledgers not found"}
        
    df_global = pd.read_csv(GLOBAL_LEDGER)
    df_core = pd.read_csv(CORE_LEDGER)
    
    report = {}
    
    for line_name, df in [("GLOBAL", df_global), ("CORE", df_core)]:
        report[line_name] = {}
        # Normalize columns
        time_col = "entry_time" if "entry_time" in df.columns else "timestamp_ny"
        df = df.sort_values(time_col).copy()
        
        for risk in RISK_LEVELS:
            results = []
            # Sample 100 random start points or every 10 trades
            for start_idx in range(0, len(df) - 20, 20):
                outcome, duration = simulate_window_challenge(df, start_idx, risk)
                results.append(outcome)
            
            total = len(results)
            passes = results.count("PASS")
            breaches = results.count("BREACH_DAILY") + results.count("BREACH_TOTAL")
            
            report[line_name][f"risk_{risk}"] = {
                "success_rate": round((passes / total) * 100, 2) if total > 0 else 0,
                "breach_rate": round((breaches / total) * 100, 2) if total > 0 else 0,
                "n_samples": total
            }
            
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    res = run_dry_run()
    print(json.dumps(res, indent=2))
