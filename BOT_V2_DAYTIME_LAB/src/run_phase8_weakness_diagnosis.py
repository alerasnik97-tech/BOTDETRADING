
import pandas as pd
import numpy as np
import json
from pathlib import Path

def diagnose_weaknesses():
    print("Phase 1: Weakness Diagnosis - PHASE7_REPAIRED_BASELINE")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\baseline_lock\phase7_repaired_baseline_trades.csv"
    trades = pd.read_csv(trades_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['year'] = trades['entry_time'].dt.year
    trades['weekday'] = trades['entry_time'].dt.day_name()
    trades['hour'] = trades['entry_time'].dt.hour
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\weakness_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance by Year
    yearly = trades.groupby('year')['r_value'].agg(['count', 'sum'])
    yearly['pf'] = trades.groupby('year').apply(lambda x: x[x['r_value']>0]['r_value'].sum() / abs(x[x['r_value']<0]['r_value'].sum()) if any(x['r_value']<0) else 0)
    yearly.to_csv(out_dir / "year_failure_analysis.csv")
    
    # 2. Performance by Level
    level_perf = trades.groupby('level')['r_value'].agg(['count', 'sum'])
    level_perf['pf'] = trades.groupby('level').apply(lambda x: x[x['r_value']>0]['r_value'].sum() / abs(x[x['r_value']<0]['r_value'].sum()) if any(x['r_value']<0) else 0)
    level_perf.to_csv(out_dir / "level_loss_breakdown.csv")
    
    # 3. Performance by Weekday
    weekday_perf = trades.groupby('weekday')['r_value'].agg(['count', 'sum'])
    weekday_perf.to_csv(out_dir / "weekday_loss_breakdown.csv")
    
    # 4. Performance by Time (Minutes post-sweep)
    trades['time_bin'] = pd.cut(trades['time_post_sweep'], bins=[0, 15, 30, 45, 60, 120])
    time_perf = trades.groupby('time_bin', observed=False)['r_value'].agg(['count', 'sum'])
    time_perf.to_csv(out_dir / "time_post_sweep_breakdown.csv")
    
    # 5. Loss Streak Analysis
    trades['is_win'] = trades['r_value'] > 0
    trades['is_loss'] = trades['r_value'] < 0
    
    streaks = []
    curr_streak = []
    for idx, row in trades.iterrows():
        if row['is_loss']:
            curr_streak.append(row)
        else:
            if len(curr_streak) >= 5:
                streaks.append(pd.DataFrame(curr_streak))
            curr_streak = []
    
    if streaks:
        all_streaks = pd.concat(streaks)
        all_streaks.to_csv(out_dir / "loss_streak_diagnosis.csv", index=False)
    
    # 6. Distance to Level / Sweep Depth
    trades['depth_bin'] = pd.cut(trades['max_depth_pips'], bins=[0, 2, 5, 10, 20, 50])
    depth_perf = trades.groupby('depth_bin', observed=False)['r_value'].agg(['count', 'sum'])
    depth_perf.to_csv(out_dir / "sweep_depth_breakdown.csv")

    diagnosis = {
        "worst_year": int(yearly['pf'].idxmin()) if not yearly.empty else "N/A",
        "worst_level": level_perf['pf'].idxmin() if not level_perf.empty else "N/A",
        "worst_weekday": weekday_perf['sum'].idxmin() if not weekday_perf.empty else "N/A",
        "avg_depth_loss": round(trades[trades['is_loss']]['max_depth_pips'].mean(), 2)
    }
    
    with open(out_dir / "weakness_diagnosis.json", 'w') as f:
        json.dump(diagnosis, f, indent=2)
        
    print("Weakness Diagnosis Complete.")

if __name__ == "__main__":
    diagnose_weaknesses()


