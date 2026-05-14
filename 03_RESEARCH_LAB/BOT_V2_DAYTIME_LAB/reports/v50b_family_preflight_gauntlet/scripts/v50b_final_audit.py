import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_family_preflight_gauntlet")

def audit():
    rdf = pd.read_csv(BASE_DIR / "results" / "V50B_MASTER_RANKING.csv")
    
    # 1. Family Scoreboard
    scoreboard = rdf.groupby("family_id").agg({
        "config_id": "count",
        "PF_train": ["median", "max"],
        "PF_val": ["median", "max"],
        "Total_R": "median"
    }).reset_index()
    scoreboard.columns = ["family_id", "configs", "PF_train_med", "PF_train_max", "PF_val_med", "PF_val_max", "R_med"]
    
    # Simple combined pass count (PF_train >= 1.0 and PF_val >= 1.1)
    passing = rdf[(rdf["PF_train"] >= 1.0) & (rdf["PF_val"] >= 1.1)]
    pass_counts = passing.groupby("family_id")["config_id"].count().reindex(scoreboard["family_id"]).fillna(0)
    scoreboard["combined_pass"] = pass_counts.values
    
    scoreboard.to_csv(BASE_DIR / "results" / "V50B_FAMILY_SCOREBOARD.csv", index=False)
    
    # 2. Rowcount Audit
    rowcount = pd.DataFrame([{
        "metric": "Total Configs", "value": 600,
        "metric": "Total Trades", "value": len(pd.read_csv(BASE_DIR / "trades" / "V50B_TRADES_ALL.csv"))
    }])
    rowcount.to_csv(BASE_DIR / "audits" / "V50B_ROWCOUNT_AUDIT.csv", index=False)
    
    print("Audit Complete.")

if __name__ == "__main__":
    audit()
