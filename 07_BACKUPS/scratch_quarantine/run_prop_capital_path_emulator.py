"""
Prop Firm Challenge Emulator & Capital Path Audit
Emulates historical survival of GLOBAL and CORE under institutional archetypes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
GLOBAL_LEDGER = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
CORE_LEDGER = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
OUTPUT_REPORT = ROOT / "results" / "SCBI_PROP_CAPITAL_PATH_REPORT.json"

INITIAL_BALANCE = 100000
RISK_PER_TRADE_PCT = 0.5 # 0.5% risk per trade (standard prop approach)

ARCHETYPES = {
    "STATIC_EQUITY": {"daily_dd": 0.05, "total_dd": 0.10, "type": "static", "consistency": 0.50},
    "TRAILING_EQUITY": {"daily_dd": 0.05, "total_dd": 0.10, "type": "trailing", "consistency": 0.50},
    "CONSISTENCY_HEAVY": {"daily_dd": 0.04, "total_dd": 0.08, "type": "static", "consistency": 0.33},
}

def simulate_prop_challenge(df, rules, name):
    balance = INITIAL_BALANCE
    max_balance = INITIAL_BALANCE
    daily_start_balance = INITIAL_BALANCE
    
    current_date = None
    trades_log = []
    
    status = "SURVIVED"
    fail_reason = ""
    profit_reached = False
    
    # Sort by time
    time_col = "entry_time" if "entry_time" in df.columns else "timestamp_ny"
    date_col = "session_date" if "session_date" in df.columns else "date"
    df = df.sort_values(time_col).copy()
    
    total_profit = 0
    max_single_trade_profit = 0
    
    for _, row in df.iterrows():
        date = row[date_col]
        pnl_r = row["pnl_r"]
        
        # New day reset
        if date != current_date:
            daily_start_balance = balance
            current_date = date
            
        # Trade impact
        risk_amt = balance * (RISK_PER_TRADE_PCT / 100)
        trade_pnl = pnl_r * risk_amt
        balance += trade_pnl
        total_profit += max(0, trade_pnl)
        max_single_trade_profit = max(max_single_trade_profit, trade_pnl)
        
        # Update High Water Mark
        max_balance = max(max_balance, balance)
        
        # Check Rules
        # 1. Daily DD
        if (daily_start_balance - balance) / daily_start_balance >= rules["daily_dd"]:
            status = "FAILED"
            fail_reason = "Daily DD Violation"
            break
            
        # 2. Total DD
        dd_base = INITIAL_BALANCE if rules["type"] == "static" else max_balance
        if (dd_base - balance) / dd_base >= rules["total_dd"]:
            status = "FAILED"
            fail_reason = "Total DD Violation"
            break
            
        # 3. Profit Target (10%)
        if (balance - INITIAL_BALANCE) / INITIAL_BALANCE >= 0.10:
            profit_reached = True
            
    # Consistency Check (at the end or if profit reached)
    if status == "SURVIVED" and profit_reached:
        if max_single_trade_profit > (total_profit * rules["consistency"]):
            status = "FAILED"
            fail_reason = "Consistency Violation (Profit Concentration)"
            
    return {
        "status": "PASSED" if profit_reached and status == "SURVIVED" else status,
        "fail_reason": fail_reason,
        "final_balance": round(balance, 2),
        "return_pct": round(((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100, 2),
        "max_single_pnl": round(max_single_trade_profit, 2)
    }

def run_audit():
    if not GLOBAL_LEDGER.exists() or not CORE_LEDGER.exists():
        return {"error": "Ledgers not found"}
        
    df_global = pd.read_csv(GLOBAL_LEDGER)
    df_core = pd.read_csv(CORE_LEDGER)
    
    results = {}
    for line_name, df in [("GLOBAL", df_global), ("CORE", df_core)]:
        results[line_name] = {}
        for arch_name, rules in ARCHETYPES.items():
            results[line_name][arch_name] = simulate_prop_challenge(df, rules, line_name)
            
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    res = run_audit()
    print(json.dumps(res, indent=2))
