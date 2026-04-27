
import pandas as pd
import numpy as np
from pathlib import Path
import json

def calculate_manual_performance():
    print("Calculating Manual Performance...")
    path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\manual_edge_alignment\manual_trades_normalized.csv"
    df = pd.read_csv(path)
    df['entry_time_ny'] = pd.to_datetime(df['entry_time_ny'])
    
    # Calculate R-based performance
    # We use 'rr' as a proxy for profit/loss in units of risk
    # If result is TP, profit is rr. If SL, loss is -1. If BE, 0.
    
    def get_r(row):
        if row['result'] == 'TP': return row['rr']
        if row['result'] == 'SL': return -1.0
        return 0.0
        
    df['pnl_r'] = df.apply(get_r, axis=1)
    
    gp = df[df['pnl_r'] > 0]['pnl_r'].sum()
    gl = abs(df[df['pnl_r'] < 0]['pnl_r'].sum())
    
    metrics = {
        "sample": len(df),
        "pf": round(gp/gl, 2) if gl > 0 else 0,
        "expectancy_r": round(df['pnl_r'].mean(), 3),
        "win_rate": round(len(df[df['result'] == 'TP']) / len(df) * 100, 2),
        "tp_count": int(len(df[df['result'] == 'TP'])),
        "sl_count": int(len(df[df['result'] == 'SL'])),
        "be_count": int(len(df[df['result'] == 'BE'])),
        "total_r": round(df['pnl_r'].sum(), 2)
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\manual_edge_alignment\manual_performance")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "manual_summary.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # By Year
    df['year'] = df['entry_time_ny'].dt.year
    df.groupby('year')['pnl_r'].agg(['count', 'sum', 'mean']).to_csv(out_dir / "manual_by_year.csv")
    
    # By Month
    df['month'] = df['entry_time_ny'].dt.month
    df.groupby('month')['pnl_r'].agg(['count', 'sum', 'mean']).to_csv(out_dir / "manual_by_month.csv")
    
    # By Hour
    df['hour'] = df['entry_time_ny'].dt.hour
    df.groupby('hour')['pnl_r'].agg(['count', 'sum', 'mean']).to_csv(out_dir / "manual_by_hour.csv")
    
    # By Weekday
    df['weekday'] = df['entry_time_ny'].dt.day_name()
    df.groupby('weekday')['pnl_r'].agg(['count', 'sum', 'mean']).to_csv(out_dir / "manual_by_weekday.csv")
    
    print("Performance Calculation Complete.")
    print(f"Manual PF: {metrics['pf']}")

if __name__ == "__main__":
    calculate_manual_performance()
