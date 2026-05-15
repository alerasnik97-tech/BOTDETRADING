import pandas as pd
import numpy as np
from pathlib import Path

def build_results():
    base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_limited_real_gauntlet_rerun_sw")
    trades_file = base_dir / "trades" / "V50B_RERUN_TRADES.csv"
    
    if not trades_file.exists():
        print("No trades found yet.")
        return
    
    df = pd.read_csv(trades_file)
    # Ensure month is string
    df["month"] = df["month"].astype(str)
    
    # Split Train/Val
    train_months = ["2020-03", "2021-08", "2022-05"]
    val_months = ["2023-01", "2024-04"]
    
    df_train = df[df["month"].isin(train_months)]
    df_val = df[df["month"].isin(val_months)]
    
    def calculate_metrics(group):
        n = len(group)
        if n == 0: return pd.Series({"N": 0, "PF": 0.0, "Total_R": 0.0, "WR": 0.0})
        
        pos_trades = group[group["net_r"] > 0]
        neg_trades = group[group["net_r"] < 0]
        
        gross_profit = pos_trades["net_r"].sum()
        gross_loss = abs(neg_trades["net_r"].sum())
        
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        total_r = group["net_r"].sum()
        wr = len(pos_trades) / n
        
        return pd.Series({
            "N": n,
            "PF": pf,
            "Total_R": total_r,
            "WR": wr
        })

    # Group by config
    res_train = df_train.groupby(["family_id", "config_id"]).apply(calculate_metrics).reset_index()
    res_val = df_val.groupby(["family_id", "config_id"]).apply(calculate_metrics).reset_index()
    
    # Master Ranking
    master = pd.merge(res_train, res_val, on=["family_id", "config_id"], suffixes=("_train", "_val"), how="outer").fillna(0)
    
    # Gates
    master["train_pass"] = (master["N_train"] >= 30) & (master["PF_train"] >= 1.0)
    master["val_pass"] = (master["N_val"] >= 20) & (master["PF_val"] >= 1.1)
    master["combined_pass"] = master["train_pass"] & master["val_pass"]
    
    # Sort and save
    master = master.sort_values(by="Total_R_val", ascending=False)
    master.to_csv(base_dir / "results" / "V50B_RERUN_MASTER_RANKING.csv", index=False)
    master[master["combined_pass"]].to_csv(base_dir / "results" / "V50B_RERUN_TOP20_GLOBAL.csv", index=False)
    
    print("Results built successfully.")

if __name__ == "__main__":
    build_results()
