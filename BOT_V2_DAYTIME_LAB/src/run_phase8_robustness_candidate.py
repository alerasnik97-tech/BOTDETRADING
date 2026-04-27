
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_candidate_robustness(name):
    print(f"Phase 5: Robustness Analysis - {name}")
    trades_path = rf"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\final_combinations\{name}_trades.csv"
    trades = pd.read_csv(trades_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['year'] = trades['entry_time'].dt.year
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Yearly
    yearly = trades.groupby('year')['r_value'].agg(['count', 'sum'])
    yearly['pf'] = trades.groupby('year').apply(lambda x: x[x['r_value']>0]['r_value'].sum() / abs(x[x['r_value']<0]['r_value'].sum()) if any(x['r_value']<0) else (100.0 if any(x['r_value']>0) else 1.0))
    yearly.to_csv(out_dir / f"phase8_robustness_by_year_{name}.csv")
    
    # Summary
    print(f"Robustness {name} Complete.")

if __name__ == "__main__":
    run_candidate_robustness("Candidate_B_F_Body60")


