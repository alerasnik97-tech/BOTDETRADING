import os
import json
import pandas as pd
from zoneinfo import ZoneInfo

# CONFIGURATION
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
INPUT_ROOT = os.path.join(BASE_DIR, r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56o_corrected_full")
CHECKPOINT_PATH = os.path.join(INPUT_ROOT, "PHASE56O_CORRECTED_FULL_CHECKPOINT.json")

def micro_analysis():
    if not os.path.exists(CHECKPOINT_PATH):
        print("ERROR: Checkpoint not found.")
        return

    with open(CHECKPOINT_PATH, 'r') as f:
        cp = json.load(f)

    all_trades = []
    for entry in cp['historical_progress']:
        if entry.get('status') != 'FORENSIC_COMPLETE': continue
        m_str = entry['month'].replace('-', '')
        csv_path = os.path.join(INPUT_ROOT, f"month_{m_str}", f"PHASE56O_MONTH_{m_str}_TRADE_LEVEL.csv")
        if os.path.exists(csv_path):
            all_trades.append(pd.read_csv(csv_path))
            
    df = pd.concat(all_trades, ignore_index=True)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    ny_tz = ZoneInfo("America/New_York")
    df['entry_ny'] = df['entry_time'].dt.tz_convert(ny_tz)
    
    # Filter 12:00 - 14:00
    df_lull = df[(df['entry_ny'].dt.hour >= 12) & (df['entry_ny'].dt.hour < 14)].copy()
    
    def get_bin(dt):
        hour = dt.hour
        minute = dt.minute
        if hour == 12 and minute < 30: return "Bloque A (12:00-12:30)"
        if hour == 12 and minute >= 30: return "Bloque B (12:30-13:00)"
        if hour == 13 and minute < 30: return "Bloque C (13:00-13:30)"
        if hour == 13 and minute >= 30: return "Bloque D (13:30-14:00)"
        return "Other"

    df_lull['bin'] = df_lull['entry_ny'].apply(get_bin)
    
    stats = []
    for bin_name in ["Bloque A (12:00-12:30)", "Bloque B (12:30-13:00)", "Bloque C (13:00-13:30)", "Bloque D (13:30-14:00)"]:
        b_trades = df_lull[df_lull['bin'] == bin_name]
        if b_trades.empty:
            stats.append({"bin": bin_name, "count": 0, "net_r": 0, "wr": 0, "pf": 0})
            continue
            
        wins = b_trades[b_trades['net_r'] > 0]
        net_r = b_trades['net_r'].sum()
        wr = (len(wins) / len(b_trades)) * 100
        win_r = wins['net_r'].sum()
        loss_r = abs(b_trades[b_trades['net_r'] <= 0]['net_r'].sum())
        pf = win_r / loss_r if loss_r > 0 else float('inf')
        
        stats.append({
            "bin": bin_name,
            "count": len(b_trades),
            "net_r": round(net_r, 2),
            "wr": round(wr, 2),
            "pf": round(pf, 2)
        })
    
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    micro_analysis()
