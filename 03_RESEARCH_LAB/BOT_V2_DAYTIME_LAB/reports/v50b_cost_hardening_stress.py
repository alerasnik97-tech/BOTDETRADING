import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

# Institutional Config
TRADES_FILE = Path("reports/v50b_limited_real_gauntlet_rerun_sw/trades/V50B_RERUN_TRADES.csv")
OUTPUT_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\cost_hardening_v50b_train_only_20260515_1020")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "BASELINE": {"mode": "zero", "slippage": 0.0, "comm": 0.0},
    "SLIPPAGE_05": {"mode": "zero", "slippage": 0.5, "comm": 0.0},
    "SLIPPAGE_10": {"mode": "zero", "slippage": 1.0, "comm": 0.0},
    "FTMO_COST": {"mode": "ftmo", "slippage": 0.5, "comm": 7.0},
    "STRESS_COMBO": {"mode": "ftmo", "slippage": 1.0, "comm": 10.0}
}

def calculate_net_r(row, scenario):
    gross_r = float(row["gross_r"])
    sl_pips = float(row["sl_pips"])
    if sl_pips <= 0: return gross_r
    
    slip_r = scenario["slippage"] / sl_pips
    comm_r = 0.0
    
    if scenario["mode"] == "ftmo":
        # $7 or $10 per lot. Standard lot = 100k. Pip value = $10.
        # Comm_R = Comm_USD / (SL_pips * Pip_Value)
        comm_r = scenario["comm"] / (sl_pips * 10.0)
    
    return gross_r - slip_r - comm_r

def run_stress_test():
    if not TRADES_FILE.exists():
        print(f"ERROR: Trades file not found at {TRADES_FILE}")
        return

    print(f"Loading certified trades from {TRADES_FILE}...")
    df = pd.read_csv(TRADES_FILE)
    
    # Filter only certified RunID 68fa2280
    df = df[df["run_id"] == "68fa2280"]
    if df.empty:
        print("ERROR: No trades found for RunID 68fa2280")
        return

    results = []
    
    for name, config in SCENARIOS.items():
        print(f"Processing Scenario: {name}...")
        df_scenario = df.copy()
        df_scenario["net_r_stress"] = df_scenario.apply(lambda r: calculate_net_r(r, config), axis=1)
        
        # Aggregates by family and config
        for (fid, cid), group in df_scenario.groupby(["family_id", "config_id"]):
            pos_trades = group[group["net_r_stress"] > 0]
            neg_trades = group[group["net_r_stress"] < 0]
            
            total_r = group["net_r_stress"].sum()
            gross_profit = pos_trades["net_r_stress"].sum()
            gross_loss = abs(neg_trades["net_r_stress"].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
            wr = len(pos_trades) / len(group)
            
            results.append({
                "family_id": fid,
                "config_id": cid,
                "scenario": name,
                "N": len(group),
                "PF": pf,
                "Total_R": total_r,
                "WR": wr,
                "Avg_Trade_R": total_r / len(group)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "COST_HARDENING_BY_SCENARIO.csv", index=False)
    
    # Summary by family (average of all configs in that family)
    family_summary = results_df.groupby(["family_id", "scenario"]).agg({
        "PF": "mean",
        "Total_R": "mean",
        "WR": "mean",
        "Avg_Trade_R": "mean"
    }).reset_index()
    family_summary.to_csv(OUTPUT_DIR / "COST_HARDENING_BY_FAMILY.csv", index=False)
    
    # Best config per family in FTMO_COST
    ftmo_best = results_df[results_df["scenario"] == "FTMO_COST"].sort_values("Total_R", ascending=False).groupby("family_id").head(1)
    ftmo_best.to_csv(OUTPUT_DIR / "COST_HARDENING_BEST_CONFIGS.csv", index=False)

    print(f"Stress test complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_stress_test()
