"""
SCBI_CORE Forward Phase 1 Runner (Autopilot Wrapper)

Simulates the forward execution of the CORE branch.
Separated from the Global line to ensure evidence integrity.
"""
import pandas as pd
import logging
import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.forward_telemetry_lib import (
    append_trace_rows,
    build_core_daily_trace_rows,
    build_core_stage2_lookup,
    build_core_trade_trace_rows,
)

LEDGER_PATH = ROOT / "results" / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"
HISTORICAL_TRADES = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
FORWARD_RUN_ID = os.environ.get("SCBI_FORWARD_RUN_ID", "").strip() or None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def initialize_ledger():
    if not LEDGER_PATH.exists():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        headers = ["event_id", "timestamp_ny", "level", "direction", "entry_price", "sl", "tp", "risk_pips", "exit_time", "exit_price", "pnl_r", "exit_reason", "news_blocked"]
        pd.DataFrame(columns=headers).to_csv(LEDGER_PATH, index=False)
        logging.info("Initialized new SCBI_CORE ledger.")

import argparse

def run_rehearsal(days=3, target_date=None):
    """Simulates a rehearsal or runs a specific day for Phase 1."""
    initialize_ledger()
    df_hist = pd.read_csv(HISTORICAL_TRADES)
    
    if target_date:
        rehearsal_trades = df_hist[df_hist["session_date"] == target_date].copy()
        if rehearsal_trades.empty:
            logging.info(f"No trades found for {target_date} in core branch.")
            return
    else:
        rehearsal_trades = df_hist.tail(days).copy()
    
    # Namespacing trades
    rehearsal_trades["event_id"] = rehearsal_trades.apply(lambda x: f"CORE_{x['session_date']}_{x['level']}_{x['direction']}", axis=1)
    
    # Append to ledger
    ledger = pd.read_csv(LEDGER_PATH)
    # Check for duplicates to prevent rerun contamination
    new_trades = rehearsal_trades[~rehearsal_trades["event_id"].isin(ledger["event_id"])]
    
    if not new_trades.empty:
        # Use institutional PnL (0.4 pips)
        new_trades["pnl_r"] = new_trades["pnl_r_institutional"]
        new_trades["timestamp_ny"] = new_trades["sweep_time"]
        
        # Mapping names to schema
        new_trades = new_trades.rename(columns={
            "exit_price_signal": "exit_price",
            "blocked_by_news": "news_blocked"
        })
        
        # Keep only ledger columns
        cols = ["event_id", "timestamp_ny", "level", "direction", "entry_price", "sl", "tp", "risk_pips", "exit_time", "exit_price", "pnl_r", "exit_reason", "news_blocked"]
        final_append = new_trades[cols]
        
        final_append.to_csv(LEDGER_PATH, mode='a', header=False, index=False)
        stage2_lookup = build_core_stage2_lookup()
        trade_trace_rows = build_core_trade_trace_rows(final_append, stage2_lookup, run_id=FORWARD_RUN_ID)
        append_trace_rows(trade_trace_rows)
        append_trace_rows(build_core_daily_trace_rows(trade_trace_rows, run_id=FORWARD_RUN_ID))
        logging.info(f"Appended {len(final_append)} trades to CORE ledger.")
    else:
        logging.warning("No new trades to append or date already processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()
    run_rehearsal(days=3, target_date=args.date)
