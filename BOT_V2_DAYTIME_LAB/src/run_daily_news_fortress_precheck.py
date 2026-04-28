
import pandas as pd
from datetime import datetime, timedelta, timezone
from news_fortress.news_fortress_gate import NewsFortressGate
import json
import os
from pathlib import Path

def run_precheck(target_date=None):
    if target_date is None:
        target_date = datetime.now(timezone.utc).date()
        
    print(f"Running News Fortress Pre-check for: {target_date}")
    
    news_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\research_lab\data\news\news_events.csv"
    if not os.path.exists(news_path):
        print("ERROR: News feed not found.")
        return
        
    df = pd.read_csv(news_path)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    
    # Filter for the day
    day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    
    day_events = df[(df['timestamp_utc'] >= day_start) & (df['timestamp_utc'] < day_end)]
    
    # We can still run the gate for any specific time
    gate = NewsFortressGate(df)
    
    critical_events = []
    for _, row in day_events.iterrows():
        title = row.get('event_name_normalized', row.get('event', 'unknown'))
        currency = row.get('currency', 'unknown')
        impact = str(row.get('impact_level', row.get('impact', 'unknown'))).upper()
        
        if impact == 'HIGH' or currency in ['USD', 'EUR']:
            critical_events.append({
                "time_utc": row['timestamp_utc'].isoformat(),
                "event": title,
                "currency": currency,
                "impact": impact
            })
            
    summary = {
        "date": str(target_date),
        "total_events": len(day_events),
        "critical_events_count": len(critical_events),
        "critical_events": critical_events,
        "trading_status": "PARTIAL_BLOCKS" if len(critical_events) > 0 else "ALLOW_ALL_DAY"
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\news_fortress_live_gate\daily_precheck")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = target_date.strftime("%Y%m%d")
    with open(out_dir / f"daily_news_fortress_precheck_{date_str}.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Pre-check complete. Status: {summary['trading_status']}")

if __name__ == "__main__":
    run_precheck(datetime(2025, 4, 3, tzinfo=timezone.utc).date())
