
import pandas as pd
import numpy as np
import json
from pathlib import Path

def calculate_manual_performance():
    input_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\manual_normalized\manual_trades_normalized.csv"
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\manual_performance")
    
    df = pd.read_csv(input_path)
    df['timestamp_entry_ny'] = pd.to_datetime(df['timestamp_entry_ny'])
    
    total_trades = len(df)
    tp_trades = df[df['result'] == 'TP']
    sl_trades = df[df['result'] == 'SL']
    be_trades = df[df['result'] == 'BE']
    
    win_rate = len(tp_trades) / total_trades if total_trades > 0 else 0
    profits = tp_trades['result_R'].sum()
    losses = abs(sl_trades['result_R'].sum())
    pf = profits / losses if losses > 0 else 0
    
    summary = {
        "sample_size": total_trades,
        "tp_count": len(tp_trades),
        "sl_count": len(sl_trades),
        "be_count": len(be_trades),
        "win_rate": round(win_rate, 4),
        "pf": round(pf, 2),
        "expectancy_r": round(df['result_R'].mean(), 2),
        "cumulative_r": round(df['result_R'].sum(), 2),
        "max_loss_streak": int(0) # Placeholder for streak calc
    }
    
    # Yearly breakdown
    df['year'] = df['timestamp_entry_ny'].dt.year
    yearly = df.groupby('year')['result_R'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'trades', 'sum': 'total_R', 'mean': 'avg_R'})
    yearly.to_csv(out_dir / "manual_trades_by_year.csv")
    
    # Monthly
    df['month'] = df['timestamp_entry_ny'].dt.month
    monthly = df.groupby('month')['result_R'].agg(['count', 'sum']).to_csv(out_dir / "manual_trades_by_month.csv")
    
    # Hourly
    df['hour'] = df['timestamp_entry_ny'].dt.hour
    hourly = df.groupby('hour')['result_R'].agg(['count', 'sum']).to_csv(out_dir / "manual_trades_by_hour.csv")
    
    with open(out_dir / "manual_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("Manual performance metrics calculated.")

if __name__ == "__main__":
    calculate_manual_performance()


