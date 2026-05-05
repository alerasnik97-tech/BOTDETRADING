import pandas as pd
import json
import os

OUTPUT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\batch_201506_201507"

def run_stress(year, month):
    file_path = os.path.join(OUTPUT_DIR, f"PHASE56_BATCH_{year}{month:02d}_TRADE_LEVEL.csv")
    df = pd.read_csv(file_path)
    
    base_r = df["pnl_r"].sum()
    sample = len(df)
    
    # EXTRA_COST_0.1R
    cost_01 = base_r - (sample * 0.1)
    
    # EXTRA_COST_0.2R
    cost_02 = base_r - (sample * 0.2)
    
    # ADVERSE_COMBINED: 0.2R cost
    adverse = cost_02 
    
    return {
        "year": year,
        "month": month,
        "base_R": round(base_r, 2),
        "extra_cost_0.1R": round(cost_01, 2),
        "extra_cost_0.2R": round(cost_02, 2),
        "adverse_combined": round(adverse, 2),
        "verdict": "EDGE_SURVIVES" if adverse > 0 else "EDGE_DEGRADES"
    }

def main():
    s06 = run_stress(2015, 6)
    s07 = run_stress(2015, 7)
    
    stress_results = [s06, s07]
    with open(os.path.join(OUTPUT_DIR, "PHASE56_BATCH_201506_201507_STRESS.json"), "w") as f:
        json.dump(stress_results, f, indent=2)

if __name__ == "__main__":
    main()
