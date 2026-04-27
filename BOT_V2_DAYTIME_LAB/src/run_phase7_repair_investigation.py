
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def investigate_news_violations():
    print("Investigating News Guard Violations...")
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\reproduction\reproduced_trades.csv"
    if not Path(trades_path).exists():
        print("Reproduced trades not found. Run reproduction first.")
        return

    trades = pd.read_csv(trades_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    
    # Load News
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    engine = Phase6Engine()
    
    # Pre-calculate news blocks
    news_blocked_times = set()
    # Concat all news for investigation
    news_list = []
    for p in ['period_2015_2019', 'period_2020_2026']:
        news_list.append(pd.read_csv(manifest[p]['news']))
    news_df = pd.concat(news_list)
    
    news_times = pd.to_datetime(news_df['timestamp_utc'], utc=True).dt.tz_convert(engine.tz_ny)
    for nt in news_times:
        for m in range(-30, 31):
            news_blocked_times.add((nt + timedelta(minutes=m)).replace(second=0, microsecond=0))
            
    violations = []
    for idx, trade in trades.iterrows():
        t = trade['entry_time'].replace(second=0, microsecond=0)
        if t in news_blocked_times:
            violations.append(trade)
            
    print(f"Found {len(violations)} violations.")
    if violations:
        v_df = pd.DataFrame(violations)
        v_df.to_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_repair\news_repair\news_violations_detailed.csv", index=False)
        print("Violations saved to outputs/phase7_repair/news_repair/news_violations_detailed.csv")

if __name__ == "__main__":
    from datetime import timedelta
    investigate_news_violations()


