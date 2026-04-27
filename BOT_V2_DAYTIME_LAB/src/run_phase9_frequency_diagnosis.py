
import pandas as pd
import numpy as np
import json
from pathlib import Path

def diagnose_frequency():
    print("Phase 1: Frequency Diagnosis - PHASE9")
    
    # Load Baseline Trades (Phase 7 repaired which was the starting point for Phase 8)
    # Actually, let's load Phase 7 and apply filters one by one to see attrition.
    phase7_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\baseline_lock\phase7_repaired_baseline_trades.csv"
    if not Path(phase7_path).exists():
        print("Phase 7 baseline not found.")
        return
        
    trades = pd.read_csv(phase7_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['weekday'] = trades['entry_time'].dt.day_name()
    
    total_months = (trades['entry_time'].max() - trades['entry_time'].min()).days / 30.44
    
    attrition = []
    
    # 1. Start with Phase 7 Repaired (The "Low precision" baseline)
    curr_trades = trades.copy()
    attrition.append({"filter": "Phase 7 Baseline", "count": len(curr_trades), "trades_per_month": len(curr_trades)/total_months})
    
    # 2. Add Friday Exclusion
    curr_trades = curr_trades[curr_trades['weekday'] != "Friday"]
    attrition.append({"filter": "+ Exclude Friday", "count": len(curr_trades), "trades_per_month": len(curr_trades)/total_months})
    
    # 3. Add Body 60% (This is the jump to Candidate B)
    curr_trades = curr_trades[curr_trades['body_pct'] >= 0.60]
    attrition.append({"filter": "+ Body 60%", "count": len(curr_trades), "trades_per_month": len(curr_trades)/total_months})
    
    # 4. Compare with "Candidate B" (Which is exactly step 3)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase9_frequency_expansion\frequency_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    attrition_df = pd.DataFrame(attrition)
    attrition_df.to_csv(out_dir / "filter_attrition_report.csv", index=False)
    
    # Monthly breakdown for Candidate B
    trades_b = trades[(trades['weekday'] != "Friday") & (trades['body_pct'] >= 0.60)]
    trades_b['month_year'] = trades_b['entry_time'].dt.to_period('M')
    monthly_counts = trades_b.groupby('month_year').size()
    monthly_counts.to_csv(out_dir / "monthly_trade_count.csv")
    
    print("Frequency Diagnosis Complete.")

if __name__ == "__main__":
    diagnose_frequency()


