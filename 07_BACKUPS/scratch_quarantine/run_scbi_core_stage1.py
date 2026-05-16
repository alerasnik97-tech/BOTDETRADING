"""
SCBI_CORE Stage-1 Execution Runner

Filters the baseline durability trades (2020-2025) according to the CORE specification
(London + Asia levels only) and applies the frozen institutional cost model (0.4 pips baseline).
"""
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TRADES_INPUT = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
OUTPUT_DIR = ROOT / "results" / "SCBI_CORE_STAGE1"

# Institutional Cost Model Calibration
# Original Durability script used 0.3 pips flat.
# Frozen Institutional Baseline is 0.4 pips.
# Delta = 0.1 pips.
COST_DELTA_PIPS = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_metrics(trades: pd.DataFrame, pnl_col: str = "pnl_r") -> dict:
    if trades.empty:
        return {"N": 0, "pf": 0.0, "total_r": 0.0, "expectancy": 0.0, "win_rate": 0.0, "max_dd": 0.0}

    pnls = trades[pnl_col]
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    # Max DD calculation
    equity = pnls.cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak).min()

    return {
        "N": len(trades),
        "pf": round(float(pf), 3),
        "total_r": round(float(pnls.sum()), 2),
        "expectancy": round(float(pnls.mean()), 4),
        "win_rate": round(float(len(wins) / len(trades)), 3),
        "max_dd": round(float(drawdown), 2)
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading global baseline trades from {TRADES_INPUT}")
    df = pd.read_csv(TRADES_INPUT)
    
    # 1. CORE FILTERING: Keep only London and Asia levels
    core_levels = ["london_h", "london_l", "asia_h", "asia_l"]
    core_df = df[df["level"].isin(core_levels)].copy()
    
    logging.info(f"Filtered {len(df)} global trades down to {len(core_df)} CORE trades (PDHL removed).")

    # 2. INSTITUTIONAL COST CALIBRATION
    # pnl_r_institutional = original_pnl_r - (COST_DELTA / risk_pips)
    core_df["pnl_r_institutional"] = core_df["pnl_r"] - (COST_DELTA_PIPS / core_df["risk_pips"])
    
    # 3. METRICS GENERATION
    global_metrics = compute_metrics(core_df, "pnl_r_institutional")
    
    # Yearly Stability
    core_df["year"] = pd.to_datetime(core_df["session_date"]).dt.year
    yearly_stats = {}
    for year, group in core_df.groupby("year"):
        yearly_stats[int(year)] = compute_metrics(group, "pnl_r_institutional")

    # Final Summary
    summary = {
        "metadata": {
            "strategy": "SCBI_CORE",
            "scope": "London+Asia",
            "period": "2020-2025",
            "baseline_cost_pips": 0.4,
            "cost_delta_applied": COST_DELTA_PIPS
        },
        "global": global_metrics,
        "yearly": yearly_stats
    }

    with open(OUTPUT_DIR / "core_stage1_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    core_df.to_csv(OUTPUT_DIR / "core_stage1_trades.csv", index=False)
    
    logging.info(f"Stage-1 Execution Complete. PF: {global_metrics['pf']}, Total R: {global_metrics['total_r']}")
    logging.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
