"""
Signal Drift Baseline Builder
Extracts statistical distributions from historical research ledgers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
GLOBAL_HIST = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
CORE_HIST = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"

OUTPUT_JSON = ROOT / "results" / "SCBI_SIGNAL_DRIFT_BASELINE.json"

def build_baseline(csv_path, name):
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    # Ensure date
    df["session_date"] = pd.to_datetime(df["session_date"])
    
    # 1. Frequency (Trades per week)
    df["week"] = df["session_date"].dt.to_period("W")
    weekly_counts = df.groupby("week").size()
    avg_weekly_trades = float(weekly_counts.mean())
    std_weekly_trades = float(weekly_counts.std())
    
    # 2. Level Composition
    level_col = "level" if "level" in df.columns else "sweep_level"
    level_dist = df[level_col].value_counts(normalize=True).to_dict()
    
    # 3. Directional
    dir_dist = df["direction"].value_counts(normalize=True).to_dict()
    
    # 4. Performance
    pnls = df["pnl_r"].values
    expectancy = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    
    return {
        "line": name,
        "frequency": {
            "avg_weekly": round(avg_weekly_trades, 2),
            "std_weekly": round(std_weekly_trades, 2)
        },
        "composition": {
            "levels": {k: round(v, 3) for k, v in level_dist.items()},
            "directions": {k: round(v, 3) for k, v in dir_dist.items()}
        },
        "performance": {
            "expectancy": round(expectancy, 4),
            "std_pnl": round(std_pnl, 4)
        }
    }

def main():
    baselines = {}
    
    global_b = build_baseline(GLOBAL_HIST, "SCBI_M5_GLOBAL")
    if global_b: baselines["SCBI_M5_GLOBAL"] = global_b
    
    core_b = build_baseline(CORE_HIST, "SCBI_CORE")
    if core_b: baselines["SCBI_CORE"] = core_b
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(baselines, f, indent=2)
    
    print(f"Baseline saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
