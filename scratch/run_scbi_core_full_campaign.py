"""
SCBI_CORE Full Campaign Runner

Executes the formal campaign (Dev/Val/Holdout) for the CORE branch.
Uses trades from Stage-2 and applies chronological segmentation and stress testing.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TRADES_INPUT = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
OUTPUT_DIR = ROOT / "results" / "SCBI_CORE_FULL_CAMPAIGN"

# Time Segments
DEV_END = "2022-06-30"
VAL_END = "2023-12-31"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_metrics(pnls) -> dict:
    if len(pnls) == 0:
        return {"N": 0, "pf": 0.0, "total_r": 0.0, "expectancy": 0.0, "max_dd": 0.0}
    
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
        "N": int(len(pnls)),
        "pf": round(float(pf), 3),
        "total_r": round(float(pnls.sum()), 2),
        "expectancy": round(float(pnls.mean()), 4),
        "max_dd": round(float(dd), 2)
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading trades from {TRADES_INPUT}")
    df = pd.read_csv(TRADES_INPUT)
    df["session_date"] = pd.to_datetime(df["session_date"])
    
    # Cost scenarios
    # pnl_r is 0.3 baseline.
    # institutional is 0.4 (+0.1 delta).
    # hard_stress is 1.2 (+0.9 delta).
    df["pnl_r_04"] = df["pnl_r_institutional"]
    df["pnl_r_12"] = df["pnl_r"] - (0.9 / df["risk_pips"])
    
    # Segmentation
    dev_mask = df["session_date"] <= pd.Timestamp(DEV_END)
    val_mask = (df["session_date"] > pd.Timestamp(DEV_END)) & (df["session_date"] <= pd.Timestamp(VAL_END))
    holdout_mask = df["session_date"] > pd.Timestamp(VAL_END)
    
    segments = {
        "Development": df[dev_mask],
        "Validation": df[val_mask],
        "Holdout": df[holdout_mask]
    }
    
    results = {
        "metadata": {
            "period_dev": f"2020-01 to {DEV_END}",
            "period_val": f"2022-07 to {VAL_END}",
            "period_holdout": "2024-01 to 2025-12"
        },
        "baseline_04": {
            "global": compute_metrics(df["pnl_r_04"]),
            "blocks": {name: compute_metrics(data["pnl_r_04"]) for name, data in segments.items()}
        },
        "stress_12": {
            "global": compute_metrics(df["pnl_r_12"]),
            "blocks": {name: compute_metrics(data["pnl_r_12"]) for name, data in segments.items()}
        }
    }
    
    with open(OUTPUT_DIR / "core_full_campaign_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logging.info("Full Campaign Execution Complete.")
    logging.info(f"PF Holdout (Baseline): {results['baseline_04']['blocks']['Holdout']['pf']}")
    logging.info(f"PF Global (Stress 1.2): {results['stress_12']['global']['pf']}")


if __name__ == "__main__":
    main()
