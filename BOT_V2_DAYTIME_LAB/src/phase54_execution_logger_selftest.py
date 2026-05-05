import os
import csv
import json
from phase54_execution_logger import log_execution_event, compute_execution_costs_R, FILLS_CSV, EVENTS_JSONL

def run_selftest():
    print("--- Phase 54: Execution Logger Self-Test ---")
    
    # Clean up test files if they exist (sintético)
    if FILLS_CSV.exists(): FILLS_CSV.unlink()
    if EVENTS_JSONL.exists(): EVENTS_JSONL.unlink()
    
    # 1. Simulate ENTRY
    entry_data = {
        "trade_id": "TEST_001",
        "symbol": "EURUSD",
        "direction": "LONG",
        "requested_price": 1.08500,
        "executed_price": 1.08505,
        "bid": 1.08500,
        "ask": 1.08503,
        "risk_pips": 10.0,
        "sl": 1.08400,
        "tp": 1.08640,
        "lot": 0.1
    }
    
    # Compute costs
    slip_pips, slip_r = compute_execution_costs_R(
        entry_data["direction"], 
        entry_data["requested_price"], 
        entry_data["executed_price"], 
        entry_data["risk_pips"]
    )
    entry_data["slippage_pips"] = slip_pips
    entry_data["slippage_R"] = slip_r
    entry_data["spread_pips"] = (entry_data["ask"] - entry_data["bid"]) * 10000
    
    log_execution_event("ENTRY", "FILLED", entry_data)
    print("Logged Entry.")
    
    # 2. Simulate EXIT (TIME_EXIT)
    exit_data = {
        "trade_id": "TEST_001",
        "symbol": "EURUSD",
        "direction": "SHORT", # Closing a Long
        "requested_price": 1.08600,
        "executed_price": 1.08595,
        "bid": 1.08595,
        "ask": 1.08598,
        "close_reason": "TIME_EXIT"
    }
    log_execution_event("EXIT", "FILLED", exit_data)
    print("Logged Exit.")
    
    # Verify files
    if FILLS_CSV.exists():
        with open(FILLS_CSV, "r") as f:
            lines = f.readlines()
            print(f"CSV Lines: {len(lines)}")
            if len(lines) >= 3:
                print("CSV Verification: PASS")
            else:
                print("CSV Verification: FAIL")
    
    if EVENTS_JSONL.exists():
        with open(EVENTS_JSONL, "r") as f:
            lines = f.readlines()
            print(f"JSONL Events: {len(lines)}")
            if len(lines) >= 2:
                print("JSONL Verification: PASS")
            else:
                print("JSONL Verification: FAIL")
                
    print("--- Self-Test Finished ---")

if __name__ == "__main__":
    run_selftest()
