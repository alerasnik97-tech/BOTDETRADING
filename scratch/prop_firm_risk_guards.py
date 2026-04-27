"""
Prop Firm Risk Guards - Implementation Layer

Provides DHL, Profit Concentration and Lot Size auditing.
Integrates with the Dual Orchestrator to ensure institutional compliance.
"""
import pandas as pd
import logging
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.forward_telemetry_lib import append_trace_rows, build_guard_trace_rows, daily_trace_ref

GLOBAL_LEDGER = ROOT / "results" / "SCBI_FORWARD_LEDGER.csv"
CORE_LEDGER = ROOT / "results" / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"
DAILY_LOSS_LIMIT_R = -3.0  # Safe proxy for 4-5% account loss
CONCENTRATION_THRESHOLD = 0.35
LOT_SIZE_STD_WARNING = 1.5
FORWARD_RUN_ID = os.environ.get("SCBI_FORWARD_RUN_ID", "").strip() or None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def check_daily_hard_loss(ledger_path, line_name):
    """Checks if the daily loss limit has been reached for a given line."""
    if not ledger_path.exists():
        return True, "No ledger found yet."
    
    try:
        df = pd.read_csv(ledger_path)
        if df.empty:
            return True, "Ledger empty."
        
        # Get current date from last entry or system
        # For simulation/rehearsal, we look at the last recorded date in the ledger
        if line_name == "SCBI_M5_GLOBAL":
            date_col = "session_date"
        else:
            # Core ledger session_date might be in event_id or timestamp
            # Assuming we can filter by the last date found
            df["tmp_date"] = df["timestamp_ny"].str[:10]
            date_col = "tmp_date"
            
        last_date = str(df[date_col].max())
        daily_pnl = pd.to_numeric(df[df[date_col] == last_date]["pnl_r"], errors="coerce").fillna(0.0).sum()
        
        if daily_pnl <= DAILY_LOSS_LIMIT_R:
            return False, f"DHL REACHED for {line_name}: {daily_pnl}R (Limit: {DAILY_LOSS_LIMIT_R}R)", last_date
        
        return True, f"Daily PnL for {line_name} is safe: {daily_pnl}R", last_date
    except Exception as e:
        return False, f"Error in DHL check for {line_name}: {e}", ""


def check_profit_concentration(ledger_path, line_name):
    """Checks if profit is too concentrated in a single trade."""
    if not ledger_path.exists():
        return True, 0.0
    
    try:
        df = pd.read_csv(ledger_path)
        pnls = df["pnl_r"].dropna().astype(float)
        if pnls.empty or pnls.sum() <= 0:
            return True, 0.0, "No positive profit concentration to evaluate."
        
        total_profit = pnls[pnls > 0].sum()
        max_trade_profit = pnls.max()
        
        ratio = max_trade_profit / total_profit if total_profit > 0 else 0
        
        if ratio > CONCENTRATION_THRESHOLD:
            return False, ratio, f"Profit concentration {ratio:.3f} exceeds {CONCENTRATION_THRESHOLD:.2f}"
        return True, ratio, f"Profit concentration {ratio:.3f} within threshold"
    except Exception as e:
        logging.error(f"Concentration error: {e}")
        return False, 0.0, f"Concentration error: {e}"


def audit_lot_size_uniformity(ledger_path, line_name):
    """Audits if risk per pips is consistent (proxy for lot size uniformity)."""
    if not ledger_path.exists():
        return True, 0.0
    
    try:
        df = pd.read_csv(ledger_path)
        if "risk_pips" not in df.columns or df.empty:
            return True, 0.0, "Lot size audit unavailable."
            
        # We check the standard deviation of risk_pips as a proxy
        # Since 1R is fixed, risk_pips should vary with volatility but sizing (Lot) should be consistent
        # In this lab, we assume fixed risk, so we audit if there are extreme outliers in pnl_r per trade type
        pnl_std = pd.to_numeric(df["pnl_r"], errors="coerce").std()
        if pd.isna(pnl_std):
            return True, 0.0, "Lot size audit insufficient sample."
        if float(pnl_std) > LOT_SIZE_STD_WARNING:
            return False, float(pnl_std), f"PnL std {float(pnl_std):.3f} exceeds proxy threshold {LOT_SIZE_STD_WARNING:.2f}"
        return True, float(pnl_std), f"PnL std {float(pnl_std):.3f} within proxy threshold"
    except Exception as e:
        return False, 0.0, f"Lot size audit error: {e}"


def run_all_guards(*, emit_trace: bool = True):
    """Runs all guards for both lines and returns a report."""
    report = {"status": "PASS", "run_id": FORWARD_RUN_ID or "", "details": []}
    
    lines = [
        ("SCBI_M5_GLOBAL", GLOBAL_LEDGER),
        ("SCBI_CORE", CORE_LEDGER)
    ]
    
    for name, path in lines:
        dhl_pass, dhl_msg, last_date = check_daily_hard_loss(path, name)
        conc_pass, conc_ratio, conc_reason = check_profit_concentration(path, name)
        lot_pass, lot_std, lot_reason = audit_lot_size_uniformity(path, name)
        daily_ref = f"results/SCBI_FORWARD_DAILY_STATUS.csv#session_date={last_date}" if name == "SCBI_M5_GLOBAL" and last_date else daily_trace_ref(name, last_date) if last_date else ""
        ledger_ref = f"results/SCBI_FORWARD_LEDGER.csv#session_date={last_date}" if name == "SCBI_M5_GLOBAL" and last_date else f"results/SCBI_CORE_PHASE1/core_phase1_ledger.csv#session_date={last_date}" if last_date else ""
        
        report["details"].append({
            "line": name,
            "last_session_date": last_date,
            "ledger_ref": ledger_ref,
            "daily_status_ref": daily_ref,
            "dhl_status": "PASS" if dhl_pass else "FAIL",
            "dhl_msg": dhl_msg,
            "concentration_ratio": round(conc_ratio, 3),
            "concentration_status": "PASS" if conc_pass else "WARNING",
            "concentration_reason": conc_reason,
            "lot_size_std": round(lot_std, 3),
            "lot_size_status": "PASS" if lot_pass else "WARNING",
            "lot_size_reason": lot_reason,
        })
        
        if not dhl_pass:
            report["status"] = "FAIL"
    if emit_trace:
        append_trace_rows(build_guard_trace_rows(report, run_id=FORWARD_RUN_ID))
    return report


if __name__ == "__main__":
    rep = run_all_guards()
    print(json.dumps(rep, indent=2))
