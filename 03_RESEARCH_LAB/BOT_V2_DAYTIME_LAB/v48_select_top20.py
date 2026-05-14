import pandas as pd
from pathlib import Path

OUT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v48_r1_real_factory_batched_run")

def select_top20():
    b1 = pd.read_csv(OUT / "R1_V48_BATCH1_RESULTS_SUMMARY.csv")
    b2 = pd.read_csv(OUT / "R1_V48_BATCH2_RESULTS_SUMMARY.csv")
    df = pd.concat([b1, b2])
    
    # Selection criteria: Combined PF > 1.0 and N > 20
    df["PF_avg"] = (df["PF_train"] + df["PF_val"]) / 2
    df["N_total"] = df["N_train"] + df["N_val"]
    
    top20 = df[(df["N_total"] >= 20)].sort_values(by=["PF_val", "PF_train"], ascending=False).head(20)
    top20.to_csv(OUT / "R1_V48_TOP20_CANDIDATES.csv", index=False)
    print(f"Top 20 selected. Best PF_val: {top20['PF_val'].max()}")

if __name__ == "__main__":
    select_top20()
