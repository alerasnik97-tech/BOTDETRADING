"""
SCBI_CORE Stage-2 Execution Quality Runner

Evaluates the CORE strategy under dynamic slippage and hard execution stress.
Uses the trades produced in Stage-1 as a starting point.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TRADES_INPUT = ROOT / "results" / "SCBI_CORE_STAGE1" / "core_stage1_trades.csv"
OUTPUT_DIR = ROOT / "results" / "SCBI_CORE_STAGE2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_metrics(pnls, trades_count) -> dict:
    if trades_count == 0:
        return {"N": 0, "pf": 0.0, "total_r": 0.0, "expectancy": 0.0}
    
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    return {
        "N": int(trades_count),
        "pf": round(float(pf), 3),
        "total_r": round(float(pnls.sum()), 2),
        "expectancy": round(float(pnls.mean()), 4)
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading Stage-1 CORE trades from {TRADES_INPUT}")
    df = pd.read_csv(TRADES_INPUT)
    
    # Baseline (Institutional 0.4 pips) already exists in column 'pnl_r_institutional'
    baseline_pnls = df["pnl_r_institutional"]
    
    # Scenario 1: Dynamic Slippage
    # Extra cost based on risk precision
    def get_dynamic_extra(risk):
        if risk < 5.0: return 0.5
        if risk < 10.0: return 0.2
        return 0.1

    df["dynamic_extra_cost"] = df["risk_pips"].apply(get_dynamic_extra)
    df["pnl_r_dynamic"] = df["pnl_r_institutional"] - (df["dynamic_extra_cost"] / df["risk_pips"])
    
    # Scenario 2: Hard Stress (1.0 pips total)
    # Stage-1 baseline was 0.4. We need +0.6 pips extra.
    df["pnl_r_hard_stress"] = df["pnl_r_institutional"] - (0.6 / df["risk_pips"])
    
    # Metrics
    results = {
        "baseline_institutional": compute_metrics(df["pnl_r_institutional"], len(df)),
        "dynamic_slippage": compute_metrics(df["pnl_r_dynamic"], len(df)),
        "hard_stress": compute_metrics(df["pnl_r_hard_stress"], len(df))
    }
    
    # Fragility Analysis: How many trades become losers under dynamic slippage?
    turned_losers = ((df["pnl_r_institutional"] > 0) & (df["pnl_r_dynamic"] <= 0)).sum()
    profit_erosion_dynamic = (1 - (results["dynamic_slippage"]["total_r"] / results["baseline_institutional"]["total_r"])) * 100
    profit_erosion_hard = (1 - (results["hard_stress"]["total_r"] / results["baseline_institutional"]["total_r"])) * 100

    results["fragility"] = {
        "trades_turned_losers_dynamic": int(turned_losers),
        "profit_erosion_dynamic_pct": round(profit_erosion_dynamic, 2),
        "profit_erosion_hard_pct": round(profit_erosion_hard, 2)
    }

    # Yearly breakdown for Dynamic
    df["year"] = pd.to_datetime(df["session_date"]).dt.year
    yearly_dynamic = {}
    for year, group in df.groupby("year"):
        yearly_dynamic[int(year)] = compute_metrics(group["pnl_r_dynamic"], len(group))
    
    results["yearly_dynamic"] = yearly_dynamic

    with open(OUTPUT_DIR / "core_stage2_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    df.to_csv(OUTPUT_DIR / "core_stage2_trades.csv", index=False)
    
    logging.info("Stage-2 Evaluation Complete.")
    logging.info(f"PF Dynamic: {results['dynamic_slippage']['pf']}")
    logging.info(f"PF Hard Stress: {results['hard_stress']['pf']}")
    logging.info(f"Profit Erosion Dynamic: {results['fragility']['profit_erosion_dynamic_pct']}%")


if __name__ == "__main__":
    main()
