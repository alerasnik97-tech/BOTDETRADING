
import pandas as pd
import json
from pathlib import Path

def audit_calendar():
    print("Starting News Calendar Audit...")
    path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\research_lab\data\news\news_events.csv"
    df = pd.read_csv(path)
    
    # Coverage
    df['ts_ny'] = pd.to_datetime(df['timestamp_ny'], utc=True)
    start_date = df['ts_ny'].min()
    end_date = df['ts_ny'].max()
    
    # Integrity checks
    stats = {
        "total_events": int(len(df)),
        "missing_currency": int(df['currency'].isna().sum()),
        "missing_impact": int(df['impact_level'].isna().sum()),
        "missing_timestamp_ny": int(df['timestamp_ny'].isna().sum()),
        "duplicate_events": int(df.duplicated().sum()),
        "high_impact_count": int(len(df[df['impact_level'] == 'HIGH'])),
        "start_date": str(start_date),
        "end_date": str(end_date)
    }
    
    # Family breakdown (Raw names)
    raw_names = df['event_name_normalized'].value_counts().head(20).to_dict()
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase17_news_feed_reliability\calendar_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "news_calendar_quality_report.json", 'w') as f:
        json.dump(stats, f, indent=2)
        
    # Coverage by year
    df['year'] = df['ts_ny'].dt.year
    coverage_by_year = df.groupby(['year', 'impact_level']).size().unstack().fillna(0)
    coverage_by_year.to_csv(out_dir / "news_calendar_coverage.csv")
    
    print(f"Audit Complete. Period: {start_date} to {end_date}")
    print(f"High Impact Events: {stats['high_impact_count']}")

if __name__ == "__main__":
    audit_calendar()
