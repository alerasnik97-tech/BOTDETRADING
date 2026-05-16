"""
Red Team Lab Audit Script
Active falsification and integrity hunter.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(".")
RESEARCH_GLOBAL = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
RESEARCH_CORE = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
FORWARD_GLOBAL = ROOT / "results" / "SCBI_FORWARD_LEDGER.csv"
FORWARD_CORE = ROOT / "results" / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"

def audit_ledger(path, name):
    if not path.exists():
        return f"MISSING: {name}"
    
    df = pd.read_csv(path)
    report = []
    
    # 1. Duplicates
    dups = df.duplicated().sum()
    if dups > 0: report.append(f"DUPLICATES_FOUND: {dups}")
    
    # 2. Outlier Extremes (>10R is highly suspicious for this alpha)
    outliers = df[df['pnl_r'] > 10].shape[0]
    if outliers > 0: report.append(f"EXTREME_OUTLIERS (>10R): {outliers}")
    
    # 3. Negative Edge check
    neg_pnl = df[df['pnl_r'] < -1.1].shape[0] # Should be impossible with hard SL at 1R + slippage
    if neg_pnl > 0: report.append(f"HARD_SL_VIOLATIONS (<-1.1R): {neg_pnl}")
    
    # 4. Zero PnL trades (Potential stale data or failed fills)
    zeros = df[df['pnl_r'] == 0].shape[0]
    if zeros > 5: report.append(f"EXCESSIVE_ZERO_PNL_TRADES: {zeros}")

    return report

def main():
    print("--- RED TEAM AUDIT: LEDGER INTEGRITY ---")
    files = [
        (RESEARCH_GLOBAL, "RESEARCH_GLOBAL"),
        (RESEARCH_CORE, "RESEARCH_CORE"),
        (FORWARD_GLOBAL, "FORWARD_GLOBAL"),
        (FORWARD_CORE, "FORWARD_CORE")
    ]
    
    all_issues = {}
    for path, name in files:
        issues = audit_ledger(path, name)
        all_issues[name] = issues
        print(f"{name}: {'OK' if not issues else issues}")

    # Governance check: Scoreboard parity
    try:
        sc = pd.read_csv("results/SCBI_DUAL_LINE_SCOREBOARD.csv")
        gl = pd.read_csv(FORWARD_GLOBAL)
        co = pd.read_csv(FORWARD_CORE)
        
        gl_n = gl[gl['event_type']=='PAPER_EXIT'].shape[0]
        co_n = co[co['event_id'].str.startswith('CORE_')].shape[0]
        
        sc_gl_n = sc[sc['Line']=='SCBI_M5_GLOBAL']['Sample_N'].iloc[0]
        sc_co_n = sc[sc['Line']=='SCBI_CORE']['Sample_N'].iloc[0]
        
        if gl_n != sc_gl_n: print(f"GOVERNANCE_MISMATCH: Global N is {gl_n} in ledger but {sc_gl_n} in scoreboard")
        if co_n != sc_co_n: print(f"GOVERNANCE_MISMATCH: Core N is {co_n} in ledger but {sc_co_n} in scoreboard")
    except Exception as e:
        print(f"GOVERNANCE_AUDIT_FAILED: {e}")

if __name__ == "__main__":
    main()
