import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

NY = ZoneInfo("America/New_York")
TRADES_PATH = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase27_full_historical_validation_2015_2026\validation_2015_2026_full\phase27_2015_2026_trades.csv")
DATA_PATH = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_m3\EURUSD_M3_BID_2020_2026.csv")
OUT_DIR = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase37x_a_daily_forced_close_audit")

def audit():
    df = pd.read_csv(TRADES_PATH)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
    
    # Baseline Forced Close (Friday 16:55)
    def is_friday_close(row):
        et = row['entry_time'].astimezone(NY)
        if et.weekday() == 4: # Friday
            fc = et.replace(hour=16, minute=55, second=0, microsecond=0)
            if row['exit_time'] >= fc:
                return True
        return False

    df['is_friday_fc'] = df.apply(is_friday_close, axis=1)
    
    # Shadow Forced Close (Mon-Thu 19:45 + Fri 16:55)
    def get_shadow_fc_time(dt):
        dt_ny = dt.astimezone(NY)
        if dt_ny.weekday() == 4:
            return dt_ny.replace(hour=16, minute=55, second=0, microsecond=0)
        else:
            return dt_ny.replace(hour=19, minute=45, second=0, microsecond=0)

    df['shadow_fc_time'] = df['entry_time'].apply(get_shadow_fc_time)
    
    # New affected trades: those that exit AFTER 19:45 NY on Mon-Thu
    new_affected = df[(df['exit_time'] > df['shadow_fc_time']) & (~df['is_friday_fc'])].copy()
    
    print(f"Total trades: {len(df)}")
    print(f"Existing Friday forced close: {df['is_friday_fc'].sum()}")
    print(f"New Shadow affected trades (Mon-Thu): {len(new_affected)}")
    
    # Analyze the 14 new trades
    if len(new_affected) > 0:
        print("\nNew Affected Trades Breakdown:")
        print(new_affected[['entry_time', 'exit_time', 'status', 'be_triggered']])

    # Load M3 data to find prices for the 14 trades
    m3 = pd.read_csv(DATA_PATH)
    m3['timestamp'] = pd.to_datetime(m3['timestamp'], utc=True)
    m3.set_index('timestamp', inplace=True)
    
    def get_price_at(dt):
        try:
            idx = m3.index.get_indexer([dt], method='pad')[0]
            if idx == -1: return None
            return m3.iloc[idx]['close']
        except: return None

    # Calculate R impact for new affected trades
    results = []
    for idx, row in new_affected.iterrows():
        fc_price = get_price_at(row['shadow_fc_time'])
        if fc_price is None: continue
            
        # Original R
        if row['status'] == 'TP': orig_r = 1.4
        elif row['status'] == 'SL' and row['be_triggered']: orig_r = 0.0
        elif row['status'] == 'SL': orig_r = -1.0
        elif row['status'] == 'FORCED_CLOSE': 
            # If it was FORCED_CLOSE in baseline, calculate its actual R
            is_long = row['type'] == 'LONG'
            risk_pips = abs(row['entry_price'] - row['sl'])
            if risk_pips < 1e-7: risk_pips = 0.0001 # fallback 1 pip
            if is_long:
                orig_r = (row['exit_price'] - row['entry_price']) / risk_pips
            else:
                orig_r = (row['entry_price'] - row['exit_price']) / risk_pips
        else: orig_r = 0.0
        
        # Shadow R
        is_long = row['type'] == 'LONG'
        risk_pips = abs(row['entry_price'] - row['sl'])
        if risk_pips < 1e-7: risk_pips = 0.0001
        
        if is_long:
            r_val = (fc_price - row['entry_price']) / risk_pips
        else:
            r_val = (row['entry_price'] - fc_price) / risk_pips
        
        results.append({
            "entry_time": row['entry_time'],
            "orig_status": row['status'],
            "orig_r": round(orig_r, 4),
            "shadow_r": round(r_val, 4),
            "delta_r": round(r_val - orig_r, 4)
        })

    shadow_df = pd.DataFrame(results)
    if not shadow_df.empty:
        print("\nImpact Analysis:")
        print(shadow_df)
        print(f"Total R Delta: {shadow_df['delta_r'].sum():.2f}")
    else:
        print("\nNo detailed impact data for the 2020-2026 period (or no trades affected in that period).")

    # Save to JSON
    summary = {
        "total_trades": len(df),
        "existing_friday_fc": int(df['is_friday_fc'].sum()),
        "new_shadow_affected": len(new_affected),
        "new_shadow_trades": results,
        "total_delta_r": float(shadow_df['delta_r'].sum()) if not shadow_df.empty else 0.0
    }
    
    with open(OUT_DIR / "shadow_audit.json", "w") as f:
        import json
        json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__":
    audit()
