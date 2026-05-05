import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Paths
MANIPULANTE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE")
LOG_DIR = MANIPULANTE / "10_LOGS_PAPER" / "ftmo_trial_bot"
FILLS_CSV = LOG_DIR / "execution_fills.csv"
EVENTS_JSONL = LOG_DIR / "execution_events.jsonl"

CSV_HEADERS = [
    "created_at_utc", "trade_id", "signal_id", "order_ticket", "symbol", "direction",
    "phase", "action", "signal_time_utc", "signal_time_ny", "request_time_utc",
    "fill_time_utc", "requested_price", "executed_price", "bid", "ask", "spread_pips",
    "sl", "tp", "risk_pips", "lot", "close_reason", "gross_R", "commission_money",
    "commission_R", "slippage_pips", "slippage_R", "net_R", "model_expected_R",
    "delta_model_vs_fill_R", "data_quality_flags", "source_file", "notes"
]

def ensure_execution_log_files():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not FILLS_CSV.exists():
        with open(FILLS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    if not EVENTS_JSONL.exists():
        EVENTS_JSONL.touch()

def _write_csv_row(row_dict: dict[str, Any]):
    try:
        ensure_execution_log_files()
        with open(FILLS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row_dict)
    except Exception as e:
        print(f"LOGGER_FAILSAFE_WARNING: Failed to write to CSV: {e}")

def _write_jsonl_event(event_dict: dict[str, Any]):
    try:
        ensure_execution_log_files()
        with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"LOGGER_FAILSAFE_WARNING: Failed to write to JSONL: {e}")

def log_execution_event(phase: str, action: str, data: dict[str, Any]):
    try:
        now_utc = datetime.now().isoformat()
        row = {k: "" for k in CSV_HEADERS}
        row["created_at_utc"] = now_utc
        row["phase"] = phase
        row["action"] = action
        
        # Update row with provided data
        for k, v in data.items():
            if k in row:
                row[k] = v
                
        _write_csv_row(row)
        
        event = {
            "timestamp_utc": now_utc,
            "phase": phase,
            "action": action,
            "payload": data
        }
        _write_jsonl_event(event)
    except Exception as e:
        print(f"LOGGER_FAILSAFE_WARNING: Unexpected logger error: {e}")

def compute_execution_costs_R(
    direction: str, 
    requested_price: float, 
    executed_price: float, 
    risk_pips: float
) -> tuple[float, float]:
    """
    Returns (slippage_pips, slippage_R)
    """
    if risk_pips <= 0:
        return 0.0, 0.0
        
    if direction == "LONG":
        # Entry slippage: executed - requested (positive is bad)
        slip_pips = (executed_price - requested_price) * 10000
    else:
        # Short entry: requested - executed (positive is bad? No, wait)
        # Entry price Short: we sell at Bid. 
        # Slippage: requested_bid - executed_bid.
        # If executed is lower than requested, slippage is positive (bad).
        slip_pips = (requested_price - executed_price) * 10000
        
    slip_r = slip_pips / risk_pips
    return round(slip_pips, 4), round(slip_r, 4)
