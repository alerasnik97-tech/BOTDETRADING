"""
SCBI Phase 1 Paper Launch Rehearsal
===================================
Simulates the daily execution of the forward test on historical dates
to certify the readiness of the automation pipeline.
"""

import os
import subprocess
import pandas as pd

ROOT = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo'
LEDGER_CSV = os.path.join(ROOT, 'results', 'SCBI_FORWARD_LEDGER.csv')
STATUS_CSV = os.path.join(ROOT, 'results', 'SCBI_FORWARD_DAILY_STATUS.csv')
RUNNER = os.path.join(ROOT, 'scratch', 'run_scbi_forward_phase1.py')

def reset_files():
    if os.path.exists(LEDGER_CSV):
        os.remove(LEDGER_CSV)
    if os.path.exists(STATUS_CSV):
        os.remove(STATUS_CSV)

def run_day(date):
    print(f"\n--- Running for date: {date} ---")
    cmd = ['python', RUNNER, '--date', date]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result

def main():
    print("Starting Rehearsal...")
    reset_files()
    
    # Test Case 1: Normal operational day (assuming valid data exists in 2024-01-05)
    res1 = run_day('2024-01-05')
    
    # Test Case 2: Re-run protection (same day again)
    res2 = run_day('2024-01-05')
    
    # Test Case 3: Data missing / insufficient data day
    res3 = run_day('2019-01-01')  # Out of range of our 2022-2025 dataset
    
    # Test Case 4: Another operational day
    res4 = run_day('2024-01-08')
    
    # Verification
    print("\n=== REHEARSAL VERIFICATION ===")
    
    # Check Re-run
    if "[BLOCKED] Target date 2024-01-05 already processed" in res2.stdout:
        print("[CHECK] Re-run protection: PASS")
    else:
        print("[CHECK] Re-run protection: FAIL")
        
    # Check Insufficient Data
    if "[DATA_ISSUE] Insufficient H1 data" in res3.stdout:
        print("[CHECK] Data issue handling: PASS")
    else:
        print("[CHECK] Data issue handling: FAIL")
        
    # Ledger checks
    if os.path.exists(LEDGER_CSV):
        ledger = pd.read_csv(LEDGER_CSV)
        print(f"[CHECK] Ledger created: PASS ({len(ledger)} rows)")
        print(ledger['event_type'].value_counts())
    else:
        print("[CHECK] Ledger created: FAIL")
        
    if os.path.exists(STATUS_CSV):
        status = pd.read_csv(STATUS_CSV)
        print(f"[CHECK] Status created: PASS ({len(status)} rows)")
        print(status[['session_date', 'sweeps_detected', 'trades_paper', 'incidents']])
    else:
        print("[CHECK] Status created: FAIL")

if __name__ == '__main__':
    main()
