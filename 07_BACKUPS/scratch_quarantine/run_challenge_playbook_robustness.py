"""
Challenge Playbook Robustness Audit (Monte Carlo & Stress)
Tests the deployment playbook under adversarial conditions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
GLOBAL_LEDGER = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
CORE_LEDGER = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
OUTPUT_REPORT = ROOT / "results" / "SCBI_CHALLENGE_PLAYBOOK_ROBUSTNESS_REPORT.json"

INITIAL_BALANCE = 100000
TARGET_PROFIT = 0.10
DAILY_LIMIT = 0.05
TOTAL_LIMIT = 0.10

def simulate_single_challenge(pnls, risk_pct):
    balance = INITIAL_BALANCE
    max_balance = INITIAL_BALANCE
    daily_pnl = 0
    
    # We assume trades happen in order of the list provided
    # For Monte Carlo, this list is already shuffled
    for pnl_r in pnls:
        # Simplification: we treat each trade as a separate day if we want to be ultra-conservative
        # or we just track total drawdown. For Monte Carlo shuffled, daily limit is harder to track 
        # without date info, so we focus on TOTAL LIMIT and a conservative Daily proxy.
        
        risk_amt = balance * (risk_pct / 100)
        trade_pnl = pnl_r * risk_amt
        balance += trade_pnl
        max_balance = max(max_balance, balance)
        
        # Total Breach
        if (INITIAL_BALANCE - balance) / INITIAL_BALANCE >= TOTAL_LIMIT:
            return "BREACH"
            
        # Success
        if (balance - INITIAL_BALANCE) / INITIAL_BALANCE >= TARGET_PROFIT:
            return "PASS"
            
    return "TIMEOUT"

def run_monte_carlo(pnls, risk_pct, iterations=1000):
    results = []
    for _ in range(iterations):
        shuffled = np.random.permutation(pnls)
        results.append(simulate_single_challenge(shuffled, risk_pct))
    
    return {
        "pass_rate": round((results.count("PASS") / iterations) * 100, 2),
        "breach_rate": round((results.count("BREACH") / iterations) * 100, 2),
        "timeout_rate": round((results.count("TIMEOUT") / iterations) * 100, 2)
    }

def run_audit():
    if not GLOBAL_LEDGER.exists() or not CORE_LEDGER.exists():
        return {"error": "Ledgers not found"}
        
    df_global = pd.read_csv(GLOBAL_LEDGER)
    df_core = pd.read_csv(CORE_LEDGER)
    
    audit_results = {}
    
    for line_name, df in [("GLOBAL", df_global), ("CORE", df_core)]:
        audit_results[line_name] = {}
        pnls_base = df["pnl_r"].values
        
        for risk in [0.25, 0.50]:
            # 1. Base Monte Carlo
            audit_results[line_name][f"mc_base_{risk}"] = run_monte_carlo(pnls_base, risk)
            
            # 2. Outlier Removal (Remove top 10% winners)
            threshold = np.percentile(pnls_base, 90)
            pnls_no_outliers = pnls_base[pnls_base < threshold]
            audit_results[line_name][f"mc_no_outliers_{risk}"] = run_monte_carlo(pnls_no_outliers, risk)
            
            # 3. Execution Stress (-0.1R per trade as proxy for slippage/fees)
            pnls_stressed = pnls_base - 0.1
            audit_results[line_name][f"mc_execution_stress_{risk}"] = run_monte_carlo(pnls_stressed, risk)
            
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(audit_results, f, indent=2)
        
    return audit_results

if __name__ == "__main__":
    res = run_audit()
    print(json.dumps(res, indent=2))
