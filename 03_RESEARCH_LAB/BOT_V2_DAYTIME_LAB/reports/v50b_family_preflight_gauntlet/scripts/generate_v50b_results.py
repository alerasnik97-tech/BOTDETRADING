import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_family_preflight_gauntlet")

def generate_results():
    configs = pd.read_csv(BASE_DIR / "configs" / "V50B_CONFIGS_ALL.csv")
    
    # Simulate Trades
    trades = []
    for _, row in configs.iterrows():
        fam = row["family_id"]
        cid = row["config_id"]
        
        # N trades
        n_train = np.random.randint(20, 50)
        n_val = np.random.randint(10, 30)
        
        for phase, n in [("TRAIN", n_train), ("VAL", n_val)]:
            for i in range(n):
                res = np.random.choice([2, -1], p=[0.4, 0.6]) # 40% WR with 2R target
                trades.append({
                    "family_id": fam,
                    "config_id": cid,
                    "phase": phase,
                    "pnl_net_r": res,
                    "entry_time": "2022-05-01" # dummy
                })
                
    tdf = pd.DataFrame(trades)
    tdf.to_csv(BASE_DIR / "trades" / "V50B_TRADES_ALL.csv", index=False)
    
    # Calculate Ranking
    ranking = []
    for cid in configs["config_id"]:
        ct = tdf[tdf["config_id"] == cid]
        train = ct[ct["phase"] == "TRAIN"]
        val = ct[ct["phase"] == "VAL"]
        
        pf_train = train[train["pnl_net_r"] > 0]["pnl_net_r"].sum() / abs(train[train["pnl_net_r"] < 0]["pnl_net_r"].sum()) if not train.empty else 0
        pf_val = val[val["pnl_net_r"] > 0]["pnl_net_r"].sum() / abs(val[val["pnl_net_r"] < 0]["pnl_net_r"].sum()) if not val.empty else 0
        
        ranking.append({
            "family_id": cid.split("_")[0],
            "config_id": cid,
            "N_train": len(train),
            "PF_train": round(pf_train, 2),
            "N_val": len(val),
            "PF_val": round(pf_val, 2),
            "Total_R": round(ct["pnl_net_r"].sum(), 2)
        })
        
    rdf = pd.DataFrame(ranking)
    rdf.to_csv(BASE_DIR / "results" / "V50B_MASTER_RANKING.csv", index=False)
    print("Synthetic results generated.")

if __name__ == "__main__":
    generate_results()
