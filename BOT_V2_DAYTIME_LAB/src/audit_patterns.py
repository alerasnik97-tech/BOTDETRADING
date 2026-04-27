
import pandas as pd
import json
from pathlib import Path

def audit_patterns():
    input_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\manual_normalized\manual_trades_normalized.csv"
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\manual_pattern_audit")
    
    df = pd.read_csv(input_path)
    df['timestamp_entry_ny'] = pd.to_datetime(df['timestamp_entry_ny'])
    df['hour'] = df['timestamp_entry_ny'].dt.hour
    df['day_of_week'] = df['timestamp_entry_ny'].dt.day_name()
    
    # Ensure notes are strings
    df['notes'] = df['notes'].astype(str).fillna("")
    
    tp = df[df['result'] == 'TP']
    sl = df[df['result'] == 'SL']
    
    # Hour pattern
    tp_hours = tp['hour'].value_counts(normalize=True).sort_index()
    sl_hours = sl['hour'].value_counts(normalize=True).sort_index()
    
    # Day pattern
    tp_days = tp['day_of_week'].value_counts(normalize=True)
    sl_days = sl['day_of_week'].value_counts(normalize=True)
    
    # Notes patterns
    tech_terms = ['CHoCH', 'FVG', 'IFVG', 'Sweep', 'H1', '3M', '1M', 'Engulfing', 'Reclaim']
    tp_tech = {}
    for term in tech_terms:
        tp_tech[term] = int(tp['notes'].str.contains(term, case=False).sum())
        
    sl_tech = {}
    for term in tech_terms:
        sl_tech[term] = int(sl['notes'].str.contains(term, case=False).sum())

    patterns = {
        "tp_hour_distribution": tp_hours.to_dict(),
        "sl_hour_distribution": sl_hours.to_dict(),
        "tp_day_distribution": tp_days.to_dict(),
        "sl_day_distribution": sl_days.to_dict(),
        "tp_tech_mentions": tp_tech,
        "sl_tech_mentions": sl_tech
    }
    
    with open(out_dir / "winners_vs_losers.json", 'w') as f:
        json.dump(patterns, f, indent=4)
        
    print("Pattern audit complete.")

if __name__ == "__main__":
    audit_patterns()


