"""
SCBI Cost Model Audit & Lockdown

Calculates empirical spreads from M1 High Precision Dukascopy data and evaluates
the impact of different cost assumptions on the SCBI_M5_GLOBAL strategy.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.data_loader import parse_prepared_index

OUTPUT_DIR = ROOT / "results" / "SCBI_COST_MODEL_AUDIT"
BID_FILE = ROOT / "data_precision" / "dukascopy" / "EURUSD_M1_BID.csv"
ASK_FILE = ROOT / "data_precision" / "dukascopy" / "EURUSD_M1_ASK.csv"
TRADES_FILE = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def measure_empirical_spreads() -> dict:
    logging.info("Reading high precision M1 data for empirical spread measurement...")
    # Load only 'open' price to save memory, as it is sufficient for spread measurement
    bid = pd.read_csv(BID_FILE, usecols=[0, 1], index_col=0, header=0, names=["timestamp", "open"])
    ask = pd.read_csv(ASK_FILE, usecols=[0, 1], index_col=0, header=0, names=["timestamp", "open"])

    bid.index = parse_prepared_index(bid.index)
    ask.index = parse_prepared_index(ask.index)

    # Join the two series
    df = bid.join(ask, lsuffix="_bid", rsuffix="_ask", how="inner")
    
    # EURUSD PIP size is 0.0001
    df["spread_pips"] = ((df["open_ask"] - df["open_bid"]) / 0.0001).clip(lower=0.0)

    # We add local hour explicitly
    df["hour_ny"] = df.index.hour

    # Session masks (NY time)
    # Asia: 20:00 - 02:00
    # London: 02:00 - 08:00
    # NY AM: 08:00 - 12:00
    cond_asia = (df["hour_ny"] >= 20) | (df["hour_ny"] < 2)
    cond_london = (df["hour_ny"] >= 2) & (df["hour_ny"] < 8)
    cond_ny = (df["hour_ny"] >= 8) & (df["hour_ny"] < 12)

    metrics = {}
    
    for session_name, mask in [("Asia", cond_asia), ("London", cond_london), ("NY_AM", cond_ny)]:
        session_data = df.loc[mask, "spread_pips"]
        metrics[session_name] = {
            "median_p50": round(float(session_data.median()), 3),
            "p95": round(float(session_data.quantile(0.95)), 3),
            "mean": round(float(session_data.mean()), 3)
        }

    # Global
    metrics["Global"] = {
        "median_p50": round(float(df["spread_pips"].median()), 3),
        "p95": round(float(df["spread_pips"].quantile(0.95)), 3),
        "mean": round(float(df["spread_pips"].mean()), 3)
    }

    return metrics


def run_ablations(metrics: dict) -> dict:
    logging.info("Running cost impact ablations on SCBI_M5_GLOBAL...")
    trades = pd.read_csv(TRADES_FILE)
    
    if trades.empty:
        return {}

    # Old baseline was 0.3 pips.
    # New spread delta = (New Spread - 0.3)
    
    max_p50 = max(metrics["Asia"]["median_p50"], metrics["London"]["median_p50"], metrics["NY_AM"]["median_p50"])
    realistic_baseline = max_p50 + 0.1
    
    max_p95 = max(metrics["Asia"]["p95"], metrics["London"]["p95"], metrics["NY_AM"]["p95"])
    mid_stress = max_p95
    
    hard_stress = 1.2

    configs = {
        "Old_Baseline_0.3": 0.3,
        f"Realistic_Empirical_{round(realistic_baseline, 2)}": realistic_baseline,
        f"Mid_Stress_P95_{round(mid_stress, 2)}": mid_stress,
        "Hard_Stress_1.2": hard_stress
    }

    results = {}
    for name, spread in configs.items():
        delta_pips = spread - 0.3
        
        # Pnl adjusted: original pnl_r - (delta_pips / risk_pips)
        adjusted_pnl = trades["pnl_r"] - (delta_pips / trades["risk_pips"])
        
        wins = adjusted_pnl[adjusted_pnl > 0]
        losses = adjusted_pnl[adjusted_pnl < 0]
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        results[name] = {
            "N": len(trades),
            "pf": round(float(pf), 3),
            "total_r": round(float(adjusted_pnl.sum()), 2),
            "expectancy": round(float(adjusted_pnl.mean()), 3)
        }

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = measure_empirical_spreads()
    ablations = run_ablations(metrics)

    # Decision Logic
    diff_sessions = max(metrics["Asia"]["median_p50"], metrics["London"]["median_p50"], metrics["NY_AM"]["median_p50"]) - \
                    min(metrics["Asia"]["median_p50"], metrics["London"]["median_p50"], metrics["NY_AM"]["median_p50"])

    if diff_sessions > 0.2:
        decision = "LOCK_SESSION_AWARE_COST_POLICY"
    else:
        decision = "LOCK_GLOBAL_COST_POLICY"

    out = {
        "empirical_spreads": metrics,
        "ablation_results": ablations,
        "spread_diff_across_sessions": round(float(diff_sessions), 3),
        "decision": decision
    }

    with open(OUTPUT_DIR / "cost_model_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logging.info(f"Audit completed. Decision: {decision}. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

