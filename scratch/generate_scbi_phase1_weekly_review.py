"""
SCBI Phase 1 - Weekly Review & Governance Script
================================================
Unifies Ledger Integrity Validation, Drift Monitoring, and Weekly Summarization.
Run this script at the end of each trading week.
"""

import os
import sys
import pandas as pd
from datetime import datetime

ROOT = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo'
LEDGER_CSV = os.path.join(ROOT, 'results', 'SCBI_FORWARD_LEDGER.csv')
STATUS_CSV = os.path.join(ROOT, 'results', 'SCBI_FORWARD_DAILY_STATUS.csv')
REVIEW_CSV = os.path.join(ROOT, 'results', 'SCBI_PHASE1_WEEKLY_REVIEW.csv')

# Research Baseline
BASE_PF = 2.44
BASE_EXP = 0.43
BASE_WR = 0.62

def validate_ledger():
    if not os.path.exists(LEDGER_CSV):
        return False, "Ledger missing", None
    try:
        df = pd.read_csv(LEDGER_CSV)
        
        # 1. Check for duplicates
        dupes = df[df.duplicated(subset=['signal_id', 'event_type'], keep=False)]
        if not dupes.empty:
            return False, f"Found {len(dupes)} duplicate events in ledger", df
            
        # 2. Check for missing PAPER_EXIT after PAPER_ENTRY
        entries = df[df['event_type'] == 'PAPER_ENTRY']
        exits = df[df['event_type'] == 'PAPER_EXIT']
        if len(entries) != len(exits):
            return False, f"Mismatch: {len(entries)} Entries vs {len(exits)} Exits", df
            
        return True, "Ledger intact", df
    except Exception as e:
        return False, f"Ledger parsing error: {str(e)}", None

def check_drift(exits_df):
    if len(exits_df) == 0:
        return {"n": 0, "status": "NO_DATA", "details": "No trades yet"}
        
    n = len(exits_df)
    exits_df['pnl_r'] = exits_df['pnl_r'].astype(float)
    wins = exits_df[exits_df['pnl_r'] > 0]
    losses = exits_df[exits_df['pnl_r'] <= 0]
    
    gp = wins['pnl_r'].sum()
    gl = abs(losses['pnl_r'].sum())
    
    pf = round(gp / gl, 2) if gl > 0 else 999
    exp = round(exits_df['pnl_r'].sum() / n, 2)
    wr = round(len(wins) / n, 2)
    
    # Calculate DD
    eq = 0
    peak = 0
    dd = 0
    for p in exits_df['pnl_r']:
        eq += p
        if eq > peak: peak = eq
        if eq - peak < dd: dd = eq - peak
    dd = round(dd, 2)
    
    # Flags
    flags = []
    status = "GREEN"
    
    if dd <= -15:
        flags.append("RED: Max DD reached (-15R)")
        status = "RED"
    elif dd <= -10:
        flags.append("YELLOW: High DD (-10R)")
        if status != "RED": status = "YELLOW"
        
    if n >= 10:
        if pf < 1.0:
            flags.append("RED: PF < 1.0")
            status = "RED"
        elif pf < 1.5:
            flags.append("YELLOW: PF < 1.5")
            if status != "RED": status = "YELLOW"
            
        if exp <= 0:
            flags.append("RED: Expectancy <= 0")
            status = "RED"
            
    return {
        "n": n, "pf": pf, "exp": exp, "wr": wr, "dd": dd,
        "status": status, "flags": "; ".join(flags) if flags else "None"
    }

def generate_weekly_review():
    print("=" * 60)
    print(" SCBI PHASE 1 - WEEKLY REVIEW & GOVERNANCE")
    print("=" * 60)
    
    # 1. Validate Ledger
    print("\n[1] LEDGER INTEGRITY CHECK")
    valid, msg, df = validate_ledger()
    if not valid:
        print(f"[ERROR] Ledger validation failed: {msg}")
        print("[ACTION] BLOCKED: Fix ledger manually before proceeding.")
        return
    print("[PASS] Ledger is fully coherent and intact.")
    
    # 2. Extract Data
    if os.path.exists(STATUS_CSV):
        status_df = pd.read_csv(STATUS_CSV)
        days_run = len(status_df)
        incidents = status_df[status_df['incidents'] != 'None']
    else:
        days_run = 0
        incidents = pd.DataFrame()
        
    exits = df[df['event_type'] == 'PAPER_EXIT'].copy()
    news_blocks = df[df['event_type'] == 'NEWS_BLOCKED'].copy()
    
    # 3. Check Drift
    print("\n[2] FORWARD DRIFT MONITORING")
    drift = check_drift(exits)
    print(f"Trades: {drift['n']} | PF: {drift.get('pf','N/A')} | Exp: {drift.get('exp','N/A')}R | WR: {drift.get('wr','N/A')}")
    print(f"Max DD: {drift.get('dd','N/A')}R")
    print(f"Drift Status: {drift['status']}")
    print(f"Active Flags: {drift['flags']}")
    
    # 4. Weekly Summary Output
    print("\n[3] OPERATIONAL SUMMARY")
    print(f"Valid operational days logged: {days_run}")
    print(f"News blocks triggered: {len(news_blocks)}")
    print(f"Critical Data Incidents: {len(incidents)}")
    if len(incidents) > 0:
        print("  -> Incident types:")
        print(incidents['incidents'].value_counts().to_string())
        
    # Append to Review CSV
    review_cols = ['review_date', 'days_run_total', 'trades_total', 'pf', 'exp', 'max_dd', 'status', 'flags']
    file_exists = os.path.exists(REVIEW_CSV)
    with open(REVIEW_CSV, 'a') as f:
        if not file_exists:
            f.write(','.join(review_cols) + '\n')
        row = [
            datetime.now().strftime('%Y-%m-%d'), str(days_run), str(drift['n']),
            str(drift.get('pf','')), str(drift.get('exp','')), str(drift.get('dd','')),
            drift['status'], drift['flags'].replace(',', '|')
        ]
        f.write(','.join(row) + '\n')
        
    print(f"\n[OK] Weekly review logged to {REVIEW_CSV}")
    
    # 5. Governance Decision
    print("\n[4] GOVERNANCE DECISION")
    if drift['status'] == "RED":
        print(">>> SUSPEND PHASE 1 <<<")
        print("Reason: RED flags detected. Re-evaluate architecture.")
    elif len(incidents) > (days_run * 0.2) and days_run > 5:
        print(">>> YELLOW WARNING <<<")
        print("Reason: >20% of days have incidents. Audit feed reliability.")
    else:
        if days_run >= 40 and drift['n'] >= 30 and drift['status'] == "GREEN":
            print(">>> ELIGIBLE FOR PROMOTION TO DEMO <<<")
        else:
            print(">>> CONTINUE PHASE 1 <<<")
            print("Reason: Within normal bounds. Keep accumulating evidence.")

if __name__ == '__main__':
    generate_weekly_review()
