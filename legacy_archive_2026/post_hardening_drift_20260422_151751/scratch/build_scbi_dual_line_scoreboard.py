"""
SCBI Dual Line Scoreboard Builder

Consolidates metrics from SCBI_M5_GLOBAL and SCBI_CORE forward ledgers.
Maintains absolute evidence separation while providing a comparative view.
"""
import pandas as pd
import logging
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
GLOBAL_LEDGER = ROOT / "results" / "SCBI_FORWARD_LEDGER.csv"
CORE_LEDGER = ROOT / "results" / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"
OUTPUT_SCOREBOARD = ROOT / "results" / "SCBI_DUAL_LINE_SCOREBOARD.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_metrics(pnls) -> dict:
    if pnls.empty:
        return {"N": 0, "pf": 0.0, "exp": 0.0, "max_dd": 0.0}
    
    pnls = pnls.dropna().astype(float)
    if pnls.empty:
        return {"N": 0, "pf": 0.0, "exp": 0.0, "max_dd": 0.0}

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    # Max DD
    equity = pnls.cumsum()
    peak = equity.cummax()
    dd = (equity - peak).min()

    return {
        "N": len(pnls),
        "pf": round(float(pf), 3),
        "exp": round(float(pnls.mean()), 4),
        "max_dd": round(float(dd), 2)
    }


def main():
    rows = []

    # 1. SCBI_M5_GLOBAL
    if GLOBAL_LEDGER.exists():
        df_g = pd.read_csv(GLOBAL_LEDGER)
        # Filter for completed trades (exit events with pnl)
        df_g_completed = df_g[df_g["pnl_r"].notnull()].copy()
        # The global ledger appends lines, so we might have duplicates if not careful.
        # Assuming event_type='PAPER_EXIT' or just last pnl entry per signal_id
        df_g_unique = df_g_completed.sort_values("event_timestamp").groupby("signal_id").last()
        
        metrics_g = compute_metrics(df_g_unique["pnl_r"])
        rows.append({
            "Line": "SCBI_M5_GLOBAL",
            "Sample_N": metrics_g["N"],
            "PF_Forward": metrics_g["pf"],
            "Exp_Forward": metrics_g["exp"],
            "Max_DD_R": metrics_g["max_dd"],
            "Last_Activity": df_g["session_date"].max() if not df_g.empty else "N/A"
        })
        logging.info(f"Processed GLOBAL line: PF {metrics_g['pf']}")
    else:
        logging.warning("GLOBAL ledger not found.")

    # 2. SCBI_CORE
    if CORE_LEDGER.exists():
        df_c = pd.read_csv(CORE_LEDGER)
        # Core ledger is simpler, one row per event
        df_c_completed = df_c[df_c["pnl_r"].notnull()].copy()
        
        metrics_c = compute_metrics(df_c_completed["pnl_r"])
        rows.append({
            "Line": "SCBI_CORE",
            "Sample_N": metrics_c["N"],
            "PF_Forward": metrics_c["pf"],
            "Exp_Forward": metrics_c["exp"],
            "Max_DD_R": metrics_c["max_dd"],
            "Last_Activity": df_c["timestamp_ny"].max() if not df_c.empty else "N/A"
        })
        logging.info(f"Processed CORE line: PF {metrics_c['pf']}")
    else:
        logging.warning("CORE ledger not found.")

    # Save Scoreboard
    if rows:
        scoreboard = pd.DataFrame(rows)
        scoreboard.to_csv(OUTPUT_SCOREBOARD, index=False)
        logging.info(f"Scoreboard saved to {OUTPUT_SCOREBOARD}")
    else:
        logging.error("No data found to build scoreboard.")


if __name__ == "__main__":
    main()
