
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_combinations():
    print("Phase 4: Improvement Combinations Laboratory")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\baseline_lock\phase7_repaired_baseline_trades.csv"
    trades = pd.read_csv(trades_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['weekday'] = trades['entry_time'].dt.day_name()
    
    combinations = [
        {"name": "Candidate_A_F_Depth20", "filters": {"weekday": ["Friday"], "max_depth_pips": 20.0}},
        {"name": "Candidate_B_F_Body60", "filters": {"weekday": ["Friday"], "body_pct": 0.60}},
        {"name": "Candidate_C_F_Depth20_Body60", "filters": {"weekday": ["Friday"], "max_depth_pips": 20.0, "body_pct": 0.60}},
        {"name": "Candidate_D_F_PDL_Only", "filters": {"weekday": ["Friday"], "level": "pdh"}}
    ]
    
    results = []
    
    for comb in combinations:
        df = trades.copy()
        if "weekday" in comb["filters"]:
            df = df[~df['weekday'].isin(comb["filters"]["weekday"])]
        if "max_depth_pips" in comb["filters"]:
            df = df[df['max_depth_pips'] <= comb["filters"]["max_depth_pips"]]
        if "body_pct" in comb["filters"]:
            df = df[df['body_pct'] >= comb["filters"]["body_pct"]]
        if "level" in comb["filters"]:
            df = df[df['level'] != comb["filters"]["level"]]
            
        gp = df[df['r_value'] > 0]['r_value'].sum()
        gl = abs(df[df['r_value'] < 0]['r_value'].sum())
        pf = gp / gl if gl > 0 else 0
        
        res = {
            "name": comb["name"],
            "sample": len(df),
            "pf": round(pf, 3),
            "expectancy": round(df['r_value'].mean(), 4),
            "cumulative_r": round(df['r_value'].sum(), 2)
        }
        results.append(res)
        
        # Save candidate trades for robustness check
        out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\final_combinations")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{comb['name']}_trades.csv", index=False)
        
    pd.DataFrame(results).to_csv(out_dir / "phase8_final_combinations.csv", index=False)
    print("Combinations Laboratory Complete.")

if __name__ == "__main__":
    run_combinations()


